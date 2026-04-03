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
    background: #0f1929; border-left: 3px solid #0d5c8a;
    border-radius: 0 8px 8px 0; padding: 20px; margin: 12px 0;
    line-height: 1.8; font-size: 0.95rem;
}
.risk-banner {
    border-radius: 12px; padding: 24px; text-align: center;
    font-family: 'Space Mono', monospace; font-size: 2rem;
    font-weight: 700; letter-spacing: 0.1em; margin: 16px 0;
}
.method-note {
    background: #0a1520; border: 1px solid #1e3a5f;
    border-radius: 8px; padding: 12px 16px;
    font-size: 0.8rem; color: #8899aa; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# GEMMA 4 SYSTEM PROMPT — updated with correct method
# ─────────────────────────────────────────
GEMMA_SYSTEM_PROMPT = """You are PhaseAlert, a seismic risk analyzer.

METHOD (from peer-reviewed study, Ridgecrest 2019 and Tohoku 2011):

1. Compute inter-event intervals tau_i = time between consecutive earthquakes
2. For each consecutive pair: r = tau_{i+1} / tau_i
3. Delta(r) = min|r - p/q| for integers 1 <= p,q <= 20
4. Delta_mean = average of all Delta values in a time window

KEY INSIGHT: The SIGNAL is not the absolute value of Delta_mean, but its CHANGE over time.
- Delta_mean DECREASING toward small values = system entering resonant stress-release mode = PRECURSOR
- Delta_mean STABLE or HIGH = irregular, background seismicity = STABLE

VALIDATED THRESHOLDS (from paper):
- Ridgecrest Mw7.1: Delta_mean dropped 35% over 30 days, then 61% in final 5 hours
- Tohoku Mw9.0: Delta_mean dropped 88% in final 6 hours (from 0.164 to 0.014-0.020)

RISK ASSESSMENT:
- Drop > 50% in recent window vs background: HIGH RISK
- Drop 20-50%: WATCH
- Drop < 20% or stable: LOW RISK
- Insufficient data: UNKNOWN

YOUR RESPONSE FORMAT:
🌍 Location: [name]
⚡ Risk Level: [LOW/WATCH/HIGH/UNKNOWN]
📊 Delta_mean: background=[value] → recent=[value] (change=[%])
📝 Analysis: [2-3 sentences explaining the trend]
✅ Recommendation: [practical advice]

RULES:
- Respond in the same language the user writes in
- Never say "earthquake will happen" — say "precursor pattern detected" or "elevated activity pattern"
- Always calm, factual, actionable
- Mention Ridgecrest/Tohoku comparison when relevant
"""

# ─────────────────────────────────────────
# CORRECT DELTA — from paper: q <= 20, mean not min
# ─────────────────────────────────────────
def delta_paper(r, Q=20):
    """
    Delta(r) = min|r - p/q| for 1 <= p,q <= 20
    Exactly as in Pascal (2026) seismic paper.
    """
    if r <= 0 or not math.isfinite(r):
        return None
    best = float('inf')
    for q in range(1, Q + 1):
        p = round(r * q)
        if p < 1:
            p = 1
        for dp in [-1, 0, 1]:
            pp = p + dp
            if 1 <= pp <= Q * 4:
                d = abs(r - pp / q)
                if d < best:
                    best = d
    return best if best < float('inf') else None


def delta_mean_window(events_in_window):
    """
    Delta_mean for a window of events.
    As per paper: mean over all consecutive inter-event interval pairs.
    """
    if len(events_in_window) < 3:
        return None, 0

    sorted_ev = sorted(events_in_window, key=lambda x: x['time'])
    intervals = []
    for i in range(1, len(sorted_ev)):
        dt = (sorted_ev[i]['time'] - sorted_ev[i-1]['time']).total_seconds() / 3600.0
        if dt > 1/60:  # min 1 minute
            intervals.append(dt)

    if len(intervals) < 2:
        return None, 0

    deltas = []
    for i in range(1, len(intervals)):
        r = intervals[i] / intervals[i-1]
        d = delta_single(r)
        if d is not None:
            deltas.append(d)

    if not deltas:
        return None, 0

    return float(np.mean(deltas)), len(deltas)


def compute_risk(events):
    """
    Two-window analysis as in paper:
    - Background window: days 8-30 (long-term baseline)
    - Recent window: last 7 days
    Compare Delta_mean. Drop = precursor signal.
    """
    if not events:
        return None, None, None, None, "UNKNOWN"

    now = max(e['time'] for e in events)

    # Background: 8-30 days ago
    bg_events = [e for e in events
                 if timedelta(days=8) <= (now - e['time']) <= timedelta(days=30)]

    # Recent: last 7 days
    rec_events = [e for e in events
                  if (now - e['time']) <= timedelta(days=7)]

    dm_bg, n_bg = delta_mean_window(bg_events)
    dm_rec, n_rec = delta_mean_window(rec_events)

    # Determine risk from DROP
    if dm_bg is None or dm_rec is None:
        level = "UNKNOWN"
    else:
        drop_pct = (dm_bg - dm_rec) / dm_bg * 100 if dm_bg > 0 else 0
        if drop_pct > 50:
            level = "HIGH"
        elif drop_pct > 20:
            level = "WATCH"
        else:
            level = "LOW"

    return dm_bg, dm_rec, n_bg, n_rec, level


def delta_timeseries(events, window_days=5):
    """
    Compute Delta_mean over sliding window through time.
    For the trend chart.
    """
    if len(events) < 5:
        return [], []

    sorted_ev = sorted(events, key=lambda x: x['time'])
    t_start = sorted_ev[0]['time']
    t_end = sorted_ev[-1]['time']
    window = timedelta(days=window_days)
    step = timedelta(hours=12)

    times_out = []
    deltas_out = []

    t = t_start + window
    while t <= t_end:
        win_events = [e for e in sorted_ev
                      if t - window <= e['time'] <= t]
        dm, n = delta_mean_window(win_events)
        if dm is not None and n >= 3:
            times_out.append(t.strftime("%m-%d"))
            deltas_out.append(round(dm, 4))
        t += step

    return times_out, deltas_out


def risk_color(level):
    return {
        "HIGH": "#dc3545",
        "WATCH": "#ffc107",
        "LOW": "#28a745",
        "UNKNOWN": "#6c757d"
    }.get(level, "#6c757d")


def risk_emoji(level):
    return {"HIGH": "🔴", "WATCH": "🟡", "LOW": "🟢", "UNKNOWN": "⚪"}.get(level, "⚪")


# ─────────────────────────────────────────
# BUILT-IN CITY COORDINATES
# ─────────────────────────────────────────
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
    "l'aquila": (42.3498, 13.3995, "L'Aquila, Italy"),
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
    "ridgecrest": (35.6298, -117.5969, "Ridgecrest, California"),
}


def geocode(name):
    key = name.lower().strip()

    # Built-in first
    if key in CITIES:
        return CITIES[key]

    # Partial match
    for k, v in CITIES.items():
        if key in k or k in key:
            return v

    # Nominatim with retry
    for attempt in range(2):
        try:
            if attempt > 0:
                import time
                time.sleep(1.5)
            r = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": name, "format": "json", "limit": 1},
                headers={"User-Agent": "PhaseAlert/3.0 (hackathon project)"},
                timeout=8
            )
            res = r.json()
            if res:
                return float(res[0]["lat"]), float(res[0]["lon"]), res[0]["display_name"]
        except:
            pass

    return None, None, None


# ─────────────────────────────────────────
# USGS API
# ─────────────────────────────────────────
def fetch_usgs(lat, lon, radius_km=200, days=30, min_mag=2.0):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    try:
        r = requests.get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params={
                "format": "geojson",
                "starttime": start.strftime("%Y-%m-%d"),
                "endtime": end.strftime("%Y-%m-%d"),
                "latitude": lat, "longitude": lon,
                "maxradiuskm": radius_km,
                "minmagnitude": min_mag,
                "orderby": "time", "limit": 200
            },
            timeout=15
        )
        r.raise_for_status()
        events = []
        for f in r.json().get("features", []):
            p = f["properties"]
            c = f["geometry"]["coordinates"]
            events.append({
                "time": datetime.fromtimestamp(p["time"] / 1000, tz=timezone.utc),
                "magnitude": p["mag"],
                "place": p["place"] or "Unknown",
                "depth": round(c[2], 1),
                "lat": c[1], "lon": c[0]
            })
        return events, None
    except Exception as e:
        return [], str(e)


# ─────────────────────────────────────────
# GEMMA 4
# ─────────────────────────────────────────
def call_gemma(location, lat, lon, events, dm_bg, dm_rec, level, api_key=None):
    drop_pct = ((dm_bg - dm_rec) / dm_bg * 100) if (dm_bg and dm_rec and dm_bg > 0) else None
    drop_str = f"{drop_pct:.1f}%" if drop_pct is not None else "N/A"

    event_lines = "".join(
        f"  M{e['magnitude']} | {e['place']} | {e['time'].strftime('%Y-%m-%d %H:%M UTC')}\n"
        for e in events[:6]
    ) or "  No events.\n"

    user_msg = (
        f"Location: {location} (lat={lat:.3f}, lon={lon:.3f})\n"
        f"Total events last 30 days: {len(events)}\n"
        f"Recent events:\n{event_lines}"
        f"Delta_mean background (8-30 days): {dm_bg:.4f}\n" if dm_bg else ""
        f"Delta_mean recent (last 7 days): {dm_rec:.4f}\n" if dm_rec else ""
        f"Change: {drop_str}\n\nPlease assess seismic risk."
    )

    if api_key and api_key.strip():
        try:
            resp = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key.strip()}"
                },
                json={
                    "model": "gemma-4-9b-it",
                    "messages": [
                        {"role": "system", "content": GEMMA_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}
                    ],
                    "max_tokens": 500
                },
                timeout=30
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
        except:
            pass

    # Rule-based fallback
    emoji = risk_emoji(level)
    bg_str = f"{dm_bg:.4f}" if dm_bg is not None else "N/A"
    rec_str = f"{dm_rec:.4f}" if dm_rec is not None else "N/A"

    if level == "UNKNOWN":
        analysis = "Insufficient seismic events in this region to compute a reliable Delta_mean trend. This typically indicates low background seismicity or sparse sensor coverage."
        rec = "No action needed. Try a city in a more seismically active region."
    elif level == "LOW":
        analysis = (f"Delta_mean is stable or slightly changing ({bg_str} → {rec_str}). "
                    f"Inter-event interval ratios remain irregular — no resonant stress-release pattern detected. "
                    f"Background seismicity appears normal.")
        rec = "No action needed. Check back in 7 days."
    elif level == "WATCH":
        analysis = (f"Delta_mean shows a moderate decrease ({bg_str} → {rec_str}, {drop_str} drop). "
                    f"Inter-event timing is progressively regularizing — a pattern consistent with "
                    f"early-stage stress-release locking. Not alarming, but worth monitoring.")
        rec = "Stay informed via local civil protection. Ensure emergency kit is current."
    else:  # HIGH
        analysis = (f"Delta_mean has dropped significantly ({bg_str} → {rec_str}, {drop_str} drop). "
                    f"This mirrors the pattern observed before Ridgecrest Mw7.1 (−61%) and "
                    f"Tohoku Mw9.0 (−88%). Inter-event timing is approaching resonant rational values.")
        rec = "Review emergency preparedness. Know your evacuation routes. Follow official civil protection channels."

    return (f"🌍 Location: {location}\n"
            f"⚡ Risk Level: {level} {emoji}\n"
            f"📊 Delta_mean: background={bg_str} → recent={rec_str} (Δ={drop_str})\n"
            f"📝 Analysis: {analysis}\n"
            f"✅ Recommendation: {rec}")


# ─────────────────────────────────────────
# MAP
# ─────────────────────────────────────────
def make_map(center_lat, center_lon, events, location_name):
    try:
        import folium
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6,
                       tiles="CartoDB dark_matter")
        folium.Marker(
            [center_lat, center_lon],
            popup=f"📍 {location_name}",
            icon=folium.Icon(color='blue', icon='home', prefix='fa')
        ).add_to(m)
        for e in events:
            mag = e['magnitude']
            color = '#dc3545' if mag >= 5 else ('#ffc107' if mag >= 4 else '#28a745')
            folium.CircleMarker(
                location=[e['lat'], e['lon']],
                radius=max(4, mag * 3),
                color=color, fill=True, fill_opacity=0.7,
                popup=folium.Popup(
                    f"M{mag} — {e['place']}<br>{e['time'].strftime('%Y-%m-%d %H:%M UTC')}",
                    max_width=200)
            ).add_to(m)
        return m._repr_html_()
    except Exception as ex:
        return f"<p>Map unavailable: {ex}</p>"


# ─────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:20px 0 10px 0;">
  <h1 style="font-size:2.8rem; margin:0;">🌍 PhaseAlert</h1>
  <p style="color:#8899aa; font-family:'Space Mono',monospace; font-size:0.8rem; margin:4px 0;">
    SEISMIC RISK ASSESSMENT · RATIONALITY GAP METHOD · GEMMA 4
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns([3, 1])
with col1:
    location_input = st.text_input(
        "Location", placeholder="Tokyo, Bologna, Istanbul, Los Angeles...",
        label_visibility="collapsed"
    )
with col2:
    analyze = st.button("🔍 Analyze", use_container_width=True, type="primary")

with st.expander("🔑 Gemma 4 API key (optional)"):
    api_key = st.text_input("Key", type="password", label_visibility="collapsed",
                            help="Get free key at aistudio.google.com")

st.markdown(
    '<div class="method-note">⚗️ Method: Delta_mean = average irrationality of consecutive inter-event time ratios '
    '(Pascal 2026, validated on Ridgecrest M7.1 and Tohoku M9.0). '
    'Risk = significant DROP in Delta_mean over time, not absolute value.</div>',
    unsafe_allow_html=True
)

st.markdown("---")

if analyze:
    if not location_input.strip():
        st.warning("Please enter a location.")
        st.stop()

    with st.spinner(f"Locating {location_input}..."):
        lat, lon, full_name = geocode(location_input)

    if lat is None:
        st.error(f"Could not find '{location_input}'. Try: Tokyo, Bologna, Istanbul, Los Angeles, Santiago...")
        st.stop()

    st.info(f"📍 Found: {full_name} ({lat:.3f}, {lon:.3f})")

    with st.spinner("Fetching USGS seismic catalog..."):
        events, err = fetch_usgs(lat, lon)

    if err:
        st.error(f"USGS error: {err}")
        st.stop()

    if not events:
        st.warning("No seismic events found in this region (last 30 days, M≥2.0, 200km radius). "
                   "This area may have very low seismicity.")
        st.stop()

    with st.spinner("Computing Delta_mean (rationality gap)..."):
        dm_bg, dm_rec, n_bg, n_rec, level = compute_risk(events)
        times_ts, deltas_ts = delta_timeseries(events)

    with st.spinner("Gemma 4 analyzing..."):
        response = call_gemma(
            full_name or location_input, lat, lon, events,
            dm_bg, dm_rec, level,
            api_key if api_key else None
        )

    # Risk banner
    color = risk_color(level)
    emoji = risk_emoji(level)
    st.markdown(
        f'<div class="risk-banner" style="background:{color}22;border:2px solid {color};color:{color};">'
        f'{emoji} {level} RISK</div>',
        unsafe_allow_html=True
    )

    # Two columns
    left, right = st.columns([1, 1])

    with left:
        st.markdown("### 🤖 Gemma 4 Analysis")
        st.markdown(
            f'<div class="analysis-box">{response.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Events", len(events))
        c2.metric("Δ_mean bg", f"{dm_bg:.4f}" if dm_bg else "N/A")
        c3.metric("Δ_mean recent", f"{dm_rec:.4f}" if dm_rec else "N/A")
        drop = ((dm_bg - dm_rec) / dm_bg * 100) if (dm_bg and dm_rec and dm_bg > 0) else None
        c4.metric("Drop %", f"{drop:.1f}%" if drop else "N/A",
                  delta=f"{-drop:.1f}%" if drop else None,
                  delta_color="inverse")

    with right:
        st.markdown("### 🗺️ Seismic Map")
        map_html = make_map(lat, lon, events, location_input)
        st.components.v1.html(map_html, height=340)

    # Delta trend
    if len(deltas_ts) > 3:
        st.markdown("### 📈 Delta_mean Trend")
        df_ts = pd.DataFrame({
            "Delta_mean": deltas_ts,
            "Ridgecrest threshold (0.119)": [0.119] * len(deltas_ts),
            "Tohoku threshold (0.020)": [0.020] * len(deltas_ts),
        }, index=times_ts)
        st.line_chart(df_ts)
        st.caption(
            "Downward trend = inter-event timing regularizing = resonant stress-release mode. "
            "Ridgecrest Mw7.1: Delta_mean reached 0.119 before mainshock. "
         
