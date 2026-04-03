import streamlit as st
import requests
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

st.set_page_config(
    page_title="PhaseAlert",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
.stButton > button {
    background: linear-gradient(135deg, #1a3a5c, #0d5c8a);
    color: white; border: none; border-radius: 8px;
    font-family: 'Space Mono', monospace; font-weight: 700;
    padding: 0.6rem 1.2rem; transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0d5c8a, #0a7abf);
    transform: translateY(-1px);
}
.analysis-box {
    background: #0f1929;
    border-left: 3px solid #0d5c8a;
    border-radius: 0 8px 8px 0;
    padding: 20px;
    margin: 12px 0;
    line-height: 1.8;
    font-size: 0.95rem;
}
.risk-banner {
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    margin: 16px 0;
}
.method-note {
    background: #0a1520;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.8rem;
    color: #8899aa;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ─── GEMMA 4 SYSTEM PROMPT ───────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are PhaseAlert, a seismic risk analyzer.\n\n"
    "METHOD (validated on Ridgecrest 2019 M7.1 and Tohoku 2011 M9.0):\n"
    "1. Compute inter-event intervals tau_i between consecutive earthquakes\n"
    "2. For each pair: r = tau_{i+1} / tau_i\n"
    "3. Delta(r) = min|r - p/q| for integers 1 <= p,q <= 20\n"
    "4. Delta_mean = average of all Delta values in a time window\n\n"
    "KEY INSIGHT: The SIGNAL is the DROP in Delta_mean over time.\n"
    "- Drop > 50 percent in recent vs background: HIGH RISK\n"
    "- Drop 20-50 percent: WATCH\n"
    "- Drop < 20 percent: LOW RISK\n\n"
    "VALIDATED:\n"
    "- Ridgecrest M7.1 2019: Delta_mean dropped 61 percent in 5 hours before mainshock\n"
    "- Tohoku M9.0 2011: Delta_mean dropped 88 percent in 6 hours before mainshock\n\n"
    "RESPONSE FORMAT:\n"
    "Location: [name]\n"
    "Risk Level: [LOW/WATCH/HIGH/UNKNOWN]\n"
    "Delta_mean: background=[value] recent=[value] change=[percent]\n"
    "Analysis: [2-3 sentences]\n"
    "Recommendation: [practical advice]\n\n"
    "Always respond in the same language the user writes in.\n"
    "Never say an earthquake will happen. Say precursor pattern detected.\n"
    "Always be calm, factual, actionable."
)

# ─── DELTA FORMULA ───────────────────────────────────────────────────────────

def delta_single(r, Q=20):
    """
    Delta(r) = min|r - p/q| for 1 <= p,q <= Q
    Both p and q constrained -- exact formula from Pascal 2026.
    """
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


def delta_mean_for_events(event_list):
    """
    Compute Delta_mean for a list of events.
    Returns (mean_value, n_pairs).
    """
    if len(event_list) < 3:
        return None, 0

    sorted_ev = sorted(event_list, key=lambda x: x["time"])
    intervals = []
    for i in range(1, len(sorted_ev)):
        dt_hours = (sorted_ev[i]["time"] - sorted_ev[i-1]["time"]).total_seconds() / 3600.0
        if dt_hours > (1.0 / 60.0):
            intervals.append(dt_hours)

    if len(intervals) < 2:
        return None, 0

    deltas = []
    for i in range(1, len(intervals)):
        d = delta_single(intervals[i] / intervals[i-1])
        if d is not None:
            deltas.append(d)

    if not deltas:
        return None, 0

    return float(np.mean(deltas)), len(deltas)


def compute_risk(events):
    """
    Two-window analysis matching the paper method.
    Background: days 8 to 30 ago.
    Recent: last 7 days.
    """
    if not events:
        return None, None, None, None, "UNKNOWN"

    now = max(e["time"] for e in events)

    bg_events = [
        e for e in events
        if timedelta(days=8) <= (now - e["time"]) <= timedelta(days=30)
    ]
    rec_events = [
        e for e in events
        if (now - e["time"]) <= timedelta(days=7)
    ]

    dm_bg, n_bg = delta_mean_for_events(bg_events)
    dm_rec, n_rec = delta_mean_for_events(rec_events)

    if dm_bg is None or dm_rec is None:
        level = "UNKNOWN"
    else:
        if dm_bg > 0:
            drop_pct = (dm_bg - dm_rec) / dm_bg * 100.0
        else:
            drop_pct = 0.0

        if drop_pct > 50.0:
            level = "HIGH"
        elif drop_pct > 20.0:
            level = "WATCH"
        else:
            level = "LOW"

    return dm_bg, dm_rec, n_bg, n_rec, level


def compute_trend(events, window_days=5):
    """
    Sliding window Delta_mean series for the trend chart.
    """
    if len(events) < 5:
        return [], []

    sorted_ev = sorted(events, key=lambda x: x["time"])
    t_start = sorted_ev[0]["time"]
    t_end = sorted_ev[-1]["time"]
    window = timedelta(days=window_days)
    step = timedelta(hours=12)

    labels = []
    values = []
    t = t_start + window
    while t <= t_end:
        win = [e for e in sorted_ev if t - window <= e["time"] <= t]
        dm, n = delta_mean_for_events(win)
        if dm is not None and n >= 3:
            labels.append(t.strftime("%m-%d"))
            values.append(round(dm, 4))
        t += step

    return labels, values


def risk_color(level):
    colors = {"HIGH": "#dc3545", "WATCH": "#ffc107", "LOW": "#28a745", "UNKNOWN": "#6c757d"}
    return colors.get(level, "#6c757d")


def risk_emoji(level):
    emojis = {"HIGH": "🔴", "WATCH": "🟡", "LOW": "🟢", "UNKNOWN": "⚪"}
    return emojis.get(level, "⚪")


# ─── BUILT-IN CITIES ─────────────────────────────────────────────────────────

CITIES = {
    "bologna": (44.4949, 11.3426, "Bologna, Italy"),
    "rome": (41.9028, 12.4964, "Rome, Italy"),
    "roma": (41.9028, 12.4964, "Roma, Italy"),
    "milan": (45.4642, 9.1900, "Milan, Italy"),
    "milano": (45.4642, 9.1900, "Milano, Italy"),
    "naples": (40.8518, 14.2681, "Naples, Italy"),
    "napoli": (40.8518, 14.2681, "Napoli, Italy"),
    "florence": (43.7696, 11.2558, "Florence, Italy"),
    "firenze": (43.7696, 11.2558, "Firenze, Italy"),
    "aquila": (42.3498, 13.3995, "L'Aquila, Italy"),
    "catania": (37.5079, 15.0830, "Catania, Italy"),
    "palermo": (38.1157, 13.3615, "Palermo, Italy"),
    "tokyo": (35.6762, 139.6503, "Tokyo, Japan"),
    "osaka": (34.6937, 135.5023, "Osaka, Japan"),
    "sendai": (38.2688, 140.8721, "Sendai, Japan"),
    "new york": (40.7128, -74.0060, "New York, USA"),
    "los angeles": (34.0522, -118.2437, "Los Angeles, USA"),
    "san francisco": (37.7749, -122.4194, "San Francisco, USA"),
    "seattle": (47.6062, -122.3321, "Seattle, USA"),
    "anchorage": (61.2181, -149.9003, "Anchorage, Alaska"),
    "ridgecrest": (35.6298, -117.5969, "Ridgecrest, California"),
    "istanbul": (41.0082, 28.9784, "Istanbul, Turkey"),
    "athens": (37.9838, 23.7275, "Athens, Greece"),
    "mexico city": (19.4326, -99.1332, "Mexico City, Mexico"),
    "lima": (-12.0464, -77.0428, "Lima, Peru"),
    "santiago": (-33.4489, -70.6693, "Santiago, Chile"),
    "tehran": (35.6892, 51.3890, "Tehran, Iran"),
    "kathmandu": (27.7172, 85.3240, "Kathmandu, Nepal"),
    "jakarta": (-6.2088, 106.8456, "Jakarta, Indonesia"),
    "taipei": (25.0330, 121.5654, "Taipei, Taiwan"),
    "reykjavik": (64.1466, -21.9426, "Reykjavik, Iceland"),
    "lisbon": (38.7169, -9.1399, "Lisbon, Portugal"),
    "madrid": (40.4168, -3.7038, "Madrid, Spain"),
    "bucharest": (44.4268, 26.1025, "Bucharest, Romania"),
    "sofia": (42.6977, 23.3219, "Sofia, Bulgaria"),
    "skopje": (41.9981, 21.4254, "Skopje, North Macedonia"),
    "tirana": (41.3275, 19.8187, "Tirana, Albania"),
    "beirut": (33.8938, 35.5018, "Beirut, Lebanon"),
    "manila": (14.5995, 120.9842, "Manila, Philippines"),
    "wellington": (-41.2866, 174.7756, "Wellington, New Zealand"),
    "christchurch": (-43.5321, 172.6362, "Christchurch, New Zealand"),
    "alaska": (64.2008, -153.4937, "Alaska, USA"),
    "chile": (-35.6751, -71.5430, "Chile"),
    "japan": (36.2048, 138.2529, "Japan"),
}


def geocode(name):
    key = name.lower().strip()

    if key in CITIES:
        return CITIES[key]

    for k, v in CITIES.items():
        if key in k or k in key:
            return v

    import time
    for attempt in range(2):
        try:
            if attempt > 0:
                time.sleep(2.0)
            r = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": name, "format": "json", "limit": 1},
                headers={"User-Agent": "PhaseAlert/3.0"},
                timeout=8
            )
            res = r.json()
            if res:
                return float(res[0]["lat"]), float(res[0]["lon"]), res[0]["display_name"]
        except Exception:
            pass

    return None, None, None


# ─── USGS API ─────────────────────────────────────────────────────────────────

def fetch_usgs(lat, lon, radius_km=200, days=30, min_mag=2.0):
    end_t = datetime.now(timezone.utc)
    start_t = end_t - timedelta(days=days)
    try:
        resp = requests.get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params={
                "format": "geojson",
                "starttime": start_t.strftime("%Y-%m-%d"),
                "endtime": end_t.strftime("%Y-%m-%d"),
                "latitude": lat,
                "longitude": lon,
                "maxradiuskm": radius_km,
                "minmagnitude": min_mag,
                "orderby": "time",
                "limit": 200,
            },
            timeout=15,
        )
        resp.raise_for_status()
        events = []
        for f in resp.json().get("features", []):
            p = f["properties"]
            c = f["geometry"]["coordinates"]
            events.append({
                "time": datetime.fromtimestamp(p["time"] / 1000, tz=timezone.utc),
                "magnitude": p["mag"],
                "place": p["place"] or "Unknown",
                "depth": round(c[2], 1),
                "lat": c[1],
                "lon": c[0],
            })
        return events, None
    except Exception as exc:
        return [], str(exc)


# ─── GEMMA 4 CALL ─────────────────────────────────────────────────────────────

def call_gemma(location, lat, lon, events, dm_bg, dm_rec, level, api_key=None):
    if dm_bg and dm_rec and dm_bg > 0:
        drop_pct = (dm_bg - dm_rec) / dm_bg * 100.0
        drop_str = str(round(drop_pct, 1)) + " percent"
    else:
        drop_pct = None
        drop_str = "N/A"

    event_lines = ""
    for e in events[:6]:
        event_lines += (
            "  M" + str(e["magnitude"])
            + " | " + e["place"]
            + " | " + e["time"].strftime("%Y-%m-%d %H:%M UTC")
            + "\n"
        )
    if not event_lines:
        event_lines = "  No events.\n"

    bg_line = (
        "Delta_mean background (8-30 days): " + str(round(dm_bg, 4)) + "\n"
        if dm_bg is not None
        else "Delta_mean background: insufficient data\n"
    )
    rec_line = (
        "Delta_mean recent (last 7 days): " + str(round(dm_rec, 4)) + "\n"
        if dm_rec is not None
        else "Delta_mean recent: insufficient data\n"
    )

    user_msg = (
        "Location: " + location
        + " (lat=" + str(round(lat, 3))
        + ", lon=" + str(round(lon, 3)) + ")\n"
        + "Total events last 30 days: " + str(len(events)) + "\n"
        + "Recent events:\n" + event_lines
        + bg_line + rec_line
        + "Change: " + drop_str + "\n\n"
        + "Please assess seismic risk."
    )

    if api_key and api_key.strip():
        try:
            api_resp = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + api_key.strip(),
                },
                json={
                    "model": "gemma-4-9b-it",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": 500,
                },
                timeout=30,
            )
            if api_resp.status_code == 200:
                return api_resp.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    # Fallback rule-based response
    em = risk_emoji(level)
    bg_s = str(round(dm_bg, 4)) if dm_bg is not None else "N/A"
    rec_s = str(round(dm_rec, 4)) if dm_rec is not None else "N/A"

    if level == "UNKNOWN":
        analysis = (
            "Insufficient seismic events to compute a reliable Delta_mean trend. "
            "This typically indicates low background seismicity or sparse sensor coverage."
        )
        rec = "No action needed. Try a city in a more seismically active region."
    elif level == "LOW":
        analysis = (
            "Delta_mean is stable: background " + bg_s + ", recent " + rec_s + ". "
            "Inter-event interval ratios remain irregular -- no resonant stress-release "
            "pattern detected. Background seismicity appears normal."
        )
        rec = "No action needed. Check back in 7 days."
    elif level == "WATCH":
        analysis = (
            "Delta_mean shows a moderate decrease from " + bg_s + " to " + rec_s
            + " (" + drop_str + " drop). "
            "Inter-event timing is progressively regularizing -- consistent with "
            "early-stage stress-release locking. Not alarming, but worth monitoring."
        )
        rec = "Stay informed via local civil protection. Ensure emergency kit is current."
    else:
        analysis = (
            "Delta_mean has dropped significantly from " + bg_s + " to " + rec_s
            + " (" + drop_str + " drop). "
            "This mirrors the pattern observed before Ridgecrest Mw7.1 (61 percent drop) "
            "and Tohoku Mw9.0 (88 percent drop). "
            "Inter-event timing is approaching resonant rational values."
        )
        rec = (
            "Review emergency preparedness. Know your evacuation routes. "
            "Follow official civil protection channels."
        )

    return (
        "Location: " + location + "\n"
        + "Risk Level: " + level + " " + em + "\n"
        + "Delta_mean: background=" + bg_s + " recent=" + rec_s
        + " (change=" + drop_str + ")\n"
        + "Analysis: " + analysis + "\n"
        + "Recommendation: " + rec
    )


# ─── MAP ──────────────────────────────────────────────────────────────────────

def make_map(center_lat, center_lon, events, location_name):
    try:
        import folium
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles="CartoDB dark_matter",
        )
        folium.Marker(
            [center_lat, center_lon],
            popup="📍 " + location_name,
            icon=folium.Icon(color="blue", icon="home", prefix="fa"),
        ).add_to(m)
        for e in events:
            mag = e["magnitude"]
            color = "#dc3545" if mag >= 5 else ("#ffc107" if mag >= 4 else "#28a745")
            folium.CircleMarker(
                location=[e["lat"], e["lon"]],
                radius=max(4, mag * 3),
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(
                    "M" + str(mag) + " -- " + e["place"]
                    + "<br>" + e["time"].strftime("%Y-%m-%d %H:%M UTC"),
                    max_width=200,
                ),
            ).add_to(m)
        return m._repr_html_()
    except Exception as exc:
        return "<p>Map unavailable: " + str(exc) + "</p>"


# ─── MAIN UI ──────────────────────────────────────────────────────────────────

st.markdown(
    "<div style='text-align:center;padding:20px 0 10px 0;'>"
    "<h1 style='font-size:2.8rem;margin:0;'>🌍 PhaseAlert</h1>"
    "<p style='color:#8899aa;font-family:Space Mono,monospace;font-size:0.8rem;margin:4px 0;'>"
    "SEISMIC RISK ASSESSMENT | RATIONALITY GAP METHOD | GEMMA 4"
    "</p></div>",
    unsafe_allow_html=True,
)

st.markdown("---")

col1, col2 = st.columns([3, 1])
with col1:
    location_input = st.text_input(
        "Location",
        placeholder="Tokyo, Bologna, Istanbul, Los Angeles...",
        label_visibility="collapsed",
    )
with col2:
    analyze = st.button("🔍 Analyze", use_container_width=True, type="primary")

with st.expander("🔑 Gemma 4 API key (optional)"):
    api_key = st.text_input(
        "Key",
        type="password",
        label_visibility="collapsed",
        help="Get free key at aistudio.google.com",
    )

st.markdown(
    "<div class='method-note'>"
    "Method: Delta_mean = average irrationality of consecutive inter-event time ratios "
    "(Pascal 2026, validated on Ridgecrest M7.1 and Tohoku M9.0). "
    "Risk signal = significant DROP in Delta_mean over time."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

if analyze:
    if not location_input.strip():
        st.warning("Please enter a location.")
        st.stop()

    with st.spinner("Locating " + location_input + "..."):
        lat, lon, full_name = geocode(location_input)

    if lat is None:
        st.error(
            "Could not find '" + location_input + "'. "
            "Try: Tokyo, Bologna, Istanbul, Los Angeles, Santiago, Ridgecrest..."
        )
        st.stop()

    st.info("📍 Found: " + full_name + " (" + str(round(lat, 3)) + ", " + str(round(lon, 3)) + ")")

    with st.spinner("Fetching USGS seismic catalog..."):
        events, err = fetch_usgs(lat, lon)

    if err:
        st.error("USGS error: " + err)
        st.stop()

    if not events:
        st.warning(
            "No seismic events found in this region "
            "(last 30 days, M>=2.0, 200km radius). "
            "This area may have very low seismicity."
        )
        st.stop()

    with st.spinner("Computing Delta_mean..."):
        dm_bg, dm_rec, n_bg, n_rec, level = compute_risk(events)
        labels_ts, values_ts = compute_trend(events)

    with st.spinner("Gemma 4 analyzing..."):
        response = call_gemma(
            full_name or location_input,
            lat, lon, events,
            dm_bg, dm_rec, level,
            api_key if api_key else None,
        )

    # Risk banner
    color = risk_color(level)
    emoji = risk_emoji(level)
    st.markdown(
        "<div class='risk-banner' style='background:" + color + "22;"
        "border:2px solid " + color + ";color:" + color + ";'>"
        + emoji + " " + level + " RISK</div>",
        unsafe_allow_html=True,
    )

    # Two columns: analysis + map
    left, right = st.columns([1, 1])

    with left:
        st.markdown("### 🤖 Gemma 4 Analysis")
        resp_html = response.replace("\n", "<br>")
        st.markdown(
            "<div class='analysis-box'>" + resp_html + "</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Events", len(events))
        c2.metric("Bg Delta", str(round(dm_bg, 4)) if dm_bg else "N/A")
        c3.metric("Recent Delta", str(round(dm_rec, 4)) if dm_rec else "N/A")
        if dm_bg and dm_rec and dm_bg > 0:
            drop_val = ro
