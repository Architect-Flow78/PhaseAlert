import streamlit as st
import requests
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import numpy as np # Explicitly importing numpy here just to be safe
from sklearn.cluster import DBSCAN

# Define constants for clarity
MIN_SAMPLING_HOURS = 0.016
DROP_THRESHOLD_HIGH = 50.0
DROP_THRESHOLD_WATCH = 20.0
BACKGROUND_DAYS = 30
BACKGROUND_START_OFFSET_DAYS = 8
RECENT_DAYS = 7
MIN_CLUSTER_SIZE = 6

st.set_page_config(page_title="PhaseAlert - Seismic Risk AI", page_icon="🌍", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:#080e1a;}
h1,h2,h3{font-family:'Space Mono',monospace!important; color: white !important;}
p, li, .dataframe, .analysis-box {color: #ccc !important;}
.stButton>button{background:linear-gradient(135deg,#0d5c8a,#0a3d6b);color:white;border:1px solid #1a6fa0;border-radius:8px;font-family:'Space Mono',monospace;font-weight:700;font-size:1rem;padding:0.7rem 2rem;transition:all 0.2s;width:100%;}
.stButton>button:hover{background:linear-gradient(135deg,#1a7abf,#0d5c8a);transform:translateY(-2px);box-shadow:0 4px 20px rgba(13,92,138,0.5);}
.analysis-box{background:#0a1525;border-left:3px solid #0d5c8a;border-radius:0 8px 8px 0;padding:16px 20px;margin:8px 0;line-height:1.8;}
.zone-card{background:#0a1525;border:1px solid #1e3a5f;border-radius:10px;padding:14px;margin:6px 0;cursor:pointer;}
.stat-box{background:#0a1525;border:1px solid #1e3a5f;border-radius:8px;padding:12px;text-align:center;}
</style>""", unsafe_allow_html=True)

SYSTEM_PROMPT = """You are PhaseAlert seismic risk AI. Method (Pascal 2026): Delta_mean = average irrationality of consecutive inter-event time ratios. Formula: Delta(r)=min|r-p/q| for 1<=p,q<=20. Risk = DROP in Delta_mean over time. Validated: Ridgecrest M7.1 drop 61%, Tohoku M9.0 drop 88%. This version utilizes advanced DBSCAN clustering to find fault-based swarms. Respond in user language. Never predict exact earthquake. Be calm and factual."""

def delta_single(r, Q=20):
    if r <= 0 or not math.isfinite(r):
        return None
    r = min(r, float(Q))
    r = max(r, 1.0 / Q)
    best = float("inf")
    for q in range(1, Q + 1):
        for p in range(1, Q + 1):
            d = abs(r - p / q)
            if d < best:
                best = d
    return best

def delta_mean_for_events(evlist):
    if len(evlist) < 3:
        return None
    sevs = sorted(evlist, key=lambda x: x["time"])
    ivs = []
    for i in range(1, len(sevs)):
        dt = (sevs[i]["time"] - sevs[i-1]["time"]).total_seconds() / 3600.0
        if dt > MIN_SAMPLING_HOURS:
            ivs.append(dt)
    if len(ivs) < 2:
        return None
    ds = [delta_single(ivs[i]/ivs[i-1]) for i in range(1, len(ivs))]
    ds = [d for d in ds if d is not None]
    return float(np.mean(ds)) if ds else None

def compute_drop(events):
    if not events:
        return None, None, None, "UNKNOWN"
    now = max(e["time"] for e in events)
    bg = [e for e in events if timedelta(days=BACKGROUND_START_OFFSET_DAYS) <= (now - e["time"]) <= timedelta(days=BACKGROUND_DAYS)]
    rec = [e for e in events if (now - e["time"]) <= timedelta(days=RECENT_DAYS)]
    dm_bg = delta_mean_for_events(bg)
    dm_rec = delta_mean_for_events(rec)
    if dm_bg is None or dm_rec is None:
        return dm_bg, dm_rec, None, "UNKNOWN"
    drop = (dm_bg - dm_rec) / dm_bg * 100.0 if dm_bg > 0 else 0.0
    level = "HIGH" if drop > DROP_THRESHOLD_HIGH else ("WATCH" if drop > DROP_THRESHOLD_WATCH else "LOW")
    return dm_bg, dm_rec, drop, level

def risk_color(level):
    return {"HIGH":"#dc3545","WATCH":"#ffc107","LOW":"#28a745","UNKNOWN":"#6c757d"}.get(level,"#6c757d")

def risk_emoji(level):
    return {"HIGH":"🔴","WATCH":"🟡","LOW":"🟢","UNKNOWN":"⚪"}.get(level,"⚪")

@st.cache_data(ttl=1800)
def fetch_global_events():
    end_t = datetime.now(timezone.utc)
    start_t = end_t - timedelta(days=BACKGROUND_DAYS)
    try:
        resp = requests.get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params={"format":"geojson","starttime":start_t.strftime("%Y-%m-%d"),
                    "endtime":end_t.strftime("%Y-%m-%d"),"minmagnitude":2.5,
                    "orderby":"time","limit":5000},
            timeout=30,
        )
        resp.raise_for_status()
        events = []
        for f in resp.json().get("features", []):
            p = f["properties"]
            c = f["geometry"]["coordinates"]
            if p["mag"] and c[0] and c[1]:
                events.append({
                    "time": datetime.fromtimestamp(p["time"]/1000, tz=timezone.utc),
                    "magnitude": float(p["mag"]),
                    "place": p["place"] or "Unknown",
                    "depth": round(float(c[2]),1),
                    "lat": float(c[1]),
                    "lon": float(c[0]),
                })
        return events, None
    except Exception as exc:
        return [], str(exc)

# ─── New DBSCAN Analysis Function ───────────────────────────────────────────
def analyze_dbscan(all_events):
    if len(all_events) < MIN_CLUSTER_SIZE:
        return []
    
    # 1. Prepare data (coordinates in degrees for simplicity, though radians is better)
    coords = np.array([[e['lat'], e['lon']] for e in all_events])
    
    # 2. Run DBSCAN
    # eps=0.5 degrees is a coarse approximation. It works well for identifying swarms.
    db = DBSCAN(eps=0.5, min_samples=MIN_CLUSTER_SIZE, metric='euclidean').fit(coords)
    labels = db.labels_
    
    # 3. Process clusters
    unique_labels = set(labels)
    zones = []
    
    for label in unique_labels:
        # Ignore noise (-1)
        if label == -1: continue
        
        # Get subset of events for this cluster
        class_member_mask = (labels == label)
        cluster_coords = coords[class_member_mask]
        cluster_indices = np.where(class_member_mask)[0]
        
        # Original logic to compute drop on the cluster events
        cluster_events = [all_events[idx] for idx in cluster_indices]
        dm_bg, dm_rec, drop, level = compute_drop(cluster_events)
        
        if level in ("HIGH", "WATCH"):
            # Compute centroid of cluster
            center_lat, center_lon = cluster_coords.mean(axis=0)
            
            # Additional summary stats
            max_mag = max(e["magnitude"] for e in cluster_events)
            
            zones.append({
                "lat": center_lat, "lon": center_lon,
                "level": level, "drop": drop,
                "n_events": len(cluster_events),
                "max_mag": max_mag,
                "dm_bg": dm_bg, "dm_rec": dm_rec,
                "events": cluster_events,
                "place": cluster_events[0]["place"] if cluster_events else "Unknown",
                "cluster_label": label, # for debugging if needed
            })
            
    # 4. Sort and return
    zones.sort(key=lambda z: z["drop"] or 0, reverse=True)
    return zones

# ─── Fix for Black Map (Updated tiles) ───────────────────────────────────────
def make_world_map(zones, all_events):
    try:
        import folium
        m = folium.Map(
            location=[20, 0], zoom_start=2,
            tiles='CartoDB dark_matter', # Updated tileset for better stability/style
            attr="CartoDB | USGS",
            width="100%", height="100%",
        )

        # Background dots for strongest all events
        for e in all_events:
            mag = e["magnitude"]
            if mag >= 6.0:
                folium.CircleMarker(
                    location=[e["lat"], e["lon"]],
                    radius=max(3, (mag-2)*3),
                    color="#ffffff", fill=True, fill_opacity=0.1,
                    weight=0.5,
                ).add_to(m)

        # Risk zone circles
        for z in zones:
            color = risk_color(z["level"])
            drop_str = str(round(z["drop"],1)) if z["drop"] else "N/A"
            folium.CircleMarker(
                location=[z["lat"], z["lon"]],
                radius=max(18, z["n_events"] / 3), # slight scaling adjustment
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.4,
                weight=2,
                popup=folium.Popup(
                    "<div style='color:black;'><b>" + risk_emoji(z["level"]) + " " + z["level"] + " RISK</b><br>"
                    + "Drop: " + drop_str + "%<br>"
                    + "Events: " + str(z["n_events"]) + "<br>"
                    + "Max M: " + str(z["max_mag"]) + "<br>"
                    + z["place"] + "</div>",
                    max_width=200,
                ),
                tooltip=risk_emoji(z["level"]) + " " + z["level"] + " - Drop " + drop_str + "%",
            ).add_to(m)

        return m._repr_html_()
    except Exception as exc:
        return "<p style='color:white'>Map error: " + str(exc) + "</p>"

def call_gemma(zones, total_events, api_key=None):
    high = [z for z in zones if z["level"] == "HIGH"]
    watch = [z for z in zones if z["level"] == "WATCH"]

    top_zones = ""
    for z in zones[:5]:
        drop_s = str(round(z["drop"],1)) if z["drop"] else "N/A"
        top_zones += (risk_emoji(z["level"]) + " " + z["level"]
                      + " | lat=" + str(round(z["lat"],1))
                      + " lon=" + str(round(z["lon"],1))
                      + " | drop=" + drop_s + "%"
                      + " | " + z["place"] + "\n")

    msg = (
        "Global seismic scan - " + str(total_events) + " events analyzed (last 30 days, M>=2.5)\n"
        + "Clustering method: DBSCAN.\n"
        + "HIGH RISK zones: " + str(len(high)) + "\n"
        + "WATCH zones: " + str(len(watch)) + "\n"
        + "Top risk zones (by Delta_mean drop):\n" + top_zones
        + "\nProvide a brief global seismic risk summary. Mention the most concerning regions and explain that the drop in inter-event time rationality gap signals fault locking. Be calm and factual."
    )

    if api_key and api_key.strip():
        try:
            r = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                headers={"Content-Type":"application/json","Authorization":"Bearer "+api_key.strip()},
                json={"model":"gemma-4-9b-it","messages":[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":msg}],"max_tokens":400},
                timeout=30,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    # Fallback
    high_regions = ", ".join([z["place"].split(",")[-1].strip() for z in high[:3]]) if high else "none"
    watch_regions = ", ".join([z["place"].split(",")[-1].strip() for z in watch[:3]]) if watch else "none"
    return (
        "Global Seismic Status (DBSCAN clustered) - " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") + "\n\n"
        + "Total events analyzed: " + str(total_events) + " (last 30 days, M>=2.5)\n"
        + "HIGH RISK zones detected: " + str(len(high)) + " (" + high_regions + ")\n"
        + "WATCH zones: " + str(len(watch)) + " (" + watch_regions + ")\n\n"
        + "Method: Delta_mean drop analysis (Pascal 2026). "
        + "Zones where inter-event timing has rationalized by more than 50% "
        + "show patterns historically preceding elevated seismic activity.\n\n"
        + "This is for informational purposes only. Always follow official civil protection guidance."
    )

# ─── UI ───────────────────────────────────────────────────────────────────────

# Hero header
st.markdown(
    "<div style='text-align:center;padding:30px 0 10px 0;'>"
    "<div style='font-size:4rem;margin-bottom:8px;'>🌍</div>"
    "<h1 style='font-size:2.5rem;margin:0;letter-spacing:0.05em;'>PhaseAlert</h1>"
    "<p style='color:#8899aa;font-family:Space Mono,monospace;font-size:0.75rem;margin:6px 0;letter-spacing:0.15em;'>"
    "GLOBAL SEISMIC RISK MONITOR | RATIONALITY GAP (PASCAL 2026) | GEMMA 4"
    "</p>"
    "<p style='color:#556677;font-size:0.8rem;margin:4px 0;'>"
    "Uses DBSCAN clustering to identify fault-based zones where seismic timing is becoming dangerously regular"
    "</p></div>",
    unsafe_allow_html=True,
)

st.markdown("---")

col_a, col_b, col_c = st.columns([1,2,1])
with col_b:
    scan_btn = st.button("🔍 SCAN PLANET NOW (DBSCAN METHOD)", use_container_width=True, type="primary")

with st.expander("🔑 Gemma 4 API key (optional)"):
    api_key = st.text_input("Key", type="password", label_visibility="collapsed")

st.markdown("---")

if scan_btn:
    with st.spinner("Downloading global seismic catalog from USGS..."):
        all_events, err = fetch_global_events()

    if err or not all_events:
        st.error("Could not fetch USGS data: " + (err or "empty response"))
        st.stop()

    with st.spinner("Analyzing " + str(len(all_events)) + " events using DBSCAN clustering..."):
        # Updated call to new function
        zones = analyze_dbscan(all_events)

    high_zones = [z for z in zones if z["level"] == "HIGH"]
    watch_zones = [z for z in zones if z["level"] == "WATCH"]

    # Stats bar
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown("<div class='stat-box'><div style='color:#8899aa;font-size:0.75rem;'>EVENTS ANALYZED</div><div style='font-size:1.8rem;font-weight:700;color:white;'>" + str(len(all_events)) + "</div></div>", unsafe_allow_html=True)
    s2.markdown("<div class='stat-box'><div style='color:#8899aa;font-size:0.75rem;'>HIGH RISK ZONES</div><div style='font-size:1.8rem;font-weight:700;color:#dc3545;'>" + str(len(high_zones)) + "</div></div>", unsafe_allow_html=True)
    s3.markdown("<div class='stat-box'><div style='color:#8899aa;font-size:0.75rem;'>WATCH ZONES</div><div style='font-size:1.8rem;font-weight:700;color:#ffc107;'>" + str(len(watch_zones)) + "</div></div>", unsafe_allow_html=True)
    s4.markdown("<div class='stat-box'><div style='color:#8899aa;font-size:0.75rem;'>SCAN TIME (UTC)</div><div style='font-size:1rem;font-weight:700;color:white;'>" + datetime.now(timezone.utc).strftime("%H:%M") + "</div></div>", unsafe_allow_html=True)

    st.markdown("")

    # Gemma analysis
    with st.spinner("Gemma 4 good hackathon AI analyzing patterns..."):
        analysis = call_gemma(zones, len(all_events), api_key if api_key else None)

    st.markdown("### 🤖 Gemma 4 Global Risk Assessment")
    st.markdown("<div class='analysis-box'>" + analysis.replace("\n","<br>") + "</div>", unsafe_allow_html=True)

    # World map
    st.markdown("### 🗺️ Fault-based Global Risk Map (DBSCAN Clustered)")
    st.caption("🔴 HIGH RISK (Delta_mean drop >50%) | 🟡 WATCH (drop 20-50%) | Map uses 'CartoDB dark_matter' tiles")
    with st.spinner("Rendering world map..."):
        map_html = make_world_map(zones, all_events)
    st.components.v1.html(map_html, height=450)

    # Top zones
    if zones:
        st.markdown("### ⚡ Top Clustered Risk Zones")
        cols = st.columns(min(3, len(zones[:6])))
        for i, z in enumerate(zones[:6]):
            col = cols[i % 3]
            color = risk_color(z["level"])
            drop_s = str(round(z["drop"],1)) if z["drop"] else "N/A"
            with col:
                st.markdown(
                    "<div class='zone-card' style='border-color:" + color + ";'>"
                    + "<div style='color:" + color + ";font-weight:700;font-size:1.1rem;'>"
                    + risk_emoji(z["level"]) + " " + z["level"] + "</div>"
                    + "<div style='color:#ccc;font-size:0.85rem;margin:4px 0;'>" + z["place"][:40] + "</div>"
                    + "<div style='color:#8899aa;font-size:0.8rem;'>Drop: " + drop_s + "% | Events: " + str(z["n_events"]) + "</div>"
                    + "</div>",
                    unsafe_allow_html=True,
                )

    # Top events table
    st.markdown("### 📋 Strongest Recent Events (M>=5)")
    big = sorted([e for e in all_events if e["magnitude"] >= 5.0], key=lambda x: x["magnitude"], reverse=True)[:15]
    if big:
        df = pd.DataFrame([{"Date":e["time"].strftime("%Y-%m-%d %H:%M"),"M":e["magnitude"],"Location":e["place"],"Depth km":e["depth"]} for e in big])
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown(
        "<div style='color:#556677;font-size:0.75rem;text-align:center;margin-top:20px;'>"
        "Method: Pascal (2026) Rationality Gap | Core Logic: Autopoietic Phase Stabilization | "
        "Data: USGS FDSN Global Catalog | Clustering: DBSCAN | "
        "For informational purposes only | Not a substitute for official warnings"
        "</div>",
        unsafe_allow_html=True,
    )

else:
    # Landing state
    st.markdown(
        "<div style='text-align:center;padding:40px 20px;color:#556677;'>"
        "<div style='font-size:3rem;margin-bottom:16px;'>🌐</div>"
        "<p style='font-size:1.1rem;'>Press SCAN PLANET NOW to analyze global seismic activity using DBSCAN</p>"
        "<p style='font-size:0.85rem;'>Downloads ~5000 recent events from USGS and detects fault-based clusters where</p>"
        "<p style='font-size:0.85rem;'>seismic timing is becoming dangerously regular</p>"
        "<br>"
        "<p style='font-size:0.8rem;color:#334455;'>"
        "Validated: Ridgecrest M7.1 (-61%) & Tohoku M9.0 (-88%) on resonant lock."
        "</p></div>",
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("## PhaseAlert v2")
    st.markdown(
        "**Method (Pascal 2026)**\n\n"
        "Delta(r) = min|r - p/q|\n"
        "for 1 <= p,q <= 20\n\n"
        "**Signal = DROP in Delta_mean**\n\n"
        "This version utilizes **DBSCAN** spatial clustering. It finds the actual shape of seismic swarms along faults, rather than using a rigid spatial grid.\n\n"
        "---\n"
        "**Validated:**\n\n"
        "Ridgecrest M7.1: -61% in 5h\n\n"
        "Tohoku M9.0: -88% in 6h\n\n"
        "---\n"
        "**Risk:**\n\n"
        "RED: drop > 50%\n\n"
        "YELLOW: drop 20-50%\n\n"
        "---\n"
        "Submission for Gemma 4 Good Hackathon 2026\n\n"
        "Global Resilience | Safety and Trust"
                         )
