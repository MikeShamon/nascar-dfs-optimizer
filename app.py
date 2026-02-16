import streamlit as st
import pandas as pd
import requests
import pulp
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="NASCAR DFS Optimizer 2026", layout="wide")
st.title("🏁 NASCAR DFS Optimizer: All Tracks (2026)")
st.markdown("Live RR data • DK/FD support • Projections & PuLP optimizer")

# --- HELPERS ---
def safe_read_html(url, desc="data"):
    try:
        return pd.read_html(requests.get(url, timeout=15).text)
    except Exception as e:
        st.warning(f"Failed to load {desc} from {url}: {str(e)}")
        return []

# --- DATA ---
@st.cache_data(ttl=1800, show_spinner=False)
def get_schedule():
    url = "https://www.racing-reference.info/season-stats/2026/W"
    tables = safe_read_html(url, "2026 schedule")
    if tables and len(tables) > 0:
        df = tables[0]
        if 'Date' in df.columns and 'Track' in df.columns:
            return df[['Date', 'Race', 'Track']].dropna(how='all')
    return pd.DataFrame({'Date': [], 'Race': ['Fallback: No schedule'], 'Track': ['Unknown']})

@st.cache_data(ttl=7200, show_spinner=False)
def get_track_history(_track_name):  # _ to allow cache per track if expanded
    recent_urls = [
        "https://www.racing-reference.info/race-results/2026-01/W",  # Daytona example; update as races happen
        "https://www.racing-reference.info/race-results/2025_Daytona_500/W",
        "https://www.racing-reference.info/race-results/2025_Coke_Zero_Sugar_400/W",
    ]
    all_dfs = []
    for url in recent_urls:
        tables = safe_read_html(url, "race history")
        if tables and len(tables) > 0:
            df = tables[0]
            needed = ['Driver', 'Start', 'Finish', 'Laps', 'Led']
            if all(col in df.columns for col in needed[:3]):  # at least basics
                df = df[needed + ['Status'] if 'Status' in df.columns else needed].copy()
                all_dfs.append(df)
    if not all_dfs:
        return pd.DataFrame()
    hist = pd.concat(all_dfs, ignore_index=True)
    if 'Driver' not in hist.columns or hist.empty:
        return pd.DataFrame()
    hist['Avg_Finish'] = hist.groupby('Driver', group_keys=False)['Finish'].transform('mean')
    hist['Laps_Led_Pct'] = (
        hist.groupby('Driver', group_keys=False)['Led'].transform('mean') /
        hist.groupby('Driver', group_keys=False)['Laps'].transform('mean').replace(0, 1)
    )
    proj = hist[['Driver', 'Avg_Finish', 'Laps_Led_Pct']].drop_duplicates()
    proj['DK_Proj_Base'] = (41 - proj['Avg_Finish'].clip(1, 40)) * 1.5 + proj['Laps_Led_Pct'] * 25
    return proj

def get_track_type(track):
    if not isinstance(track, str): return 'intermediate'
    superspeed = ['Daytona', 'Talladega']
    if any(s in track for s in superspeed): return 'superspeedway'
    return 'intermediate'  # simplify for now

# --- UI ---
schedule = get_schedule()
st.sidebar.header("Race Selection")
if not schedule.empty:
    race_options = [f"{r} @ {t}" for r, t in zip(schedule['Race'], schedule['Track'])]
    selected = st.sidebar.selectbox("Choose Race", race_options, index=0)
    idx = race_options.index(selected)
    race = schedule.iloc[idx]['Race']
    track = schedule.iloc[idx]['Track']
else:
    race, track = "No race data", "Unknown"
    st.sidebar.warning("Could not load schedule – using fallback mode")

track_type = get_track_type(track)
st.sidebar.success(f"**{race} @ {track}** • ({track_type}) • {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Projections
st.header("Driver Projections (Recent History)")
hist = get_track_history(track)
if not hist.empty:
    st.dataframe(hist.sort_values('DK_Proj_Base', ascending=False).head(20), use_container_width=True)
    fig = px.bar(hist.nlargest(15, 'DK_Proj_Base'), x='Driver', y='DK_Proj_Base', title="Top Projections")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No recent race history loaded – projections will use neutral defaults after upload.")

# Upload
st.header("Upload Salaries (DK or FD)")
dk_file = st.file_uploader("DraftKings CSV", type="csv")
fd_file = st.file_uploader("FanDuel CSV", type="csv")

salaries = None
salary_cap = 50000
if dk_file:
    salaries = pd.read_csv(dk_file)
    salaries = salaries.rename(columns={'Name': 'Driver', 'Roster Position': 'Position'})
    salary_cap = 50000
elif fd_file:
    salaries = pd.read_csv(fd_file)
    salaries = salaries.rename(columns={'Player Name': 'Driver'})
    salary_cap = 60000

if salaries is not None:
    st.success(f"Loaded {len(salaries)} drivers")
    salaries = salaries.merge(hist, on='Driver', how='left').fillna({'DK_Proj_Base': 20, 'Avg_Finish': 20})
    mult = 1.3 if track_type == 'superspeedway' else 1.0
    salaries['Final_Proj'] = salaries['DK_Proj_Base'] * mult
    st.dataframe(salaries[['Driver', 'Salary', 'Final_Proj']].sort_values('Final_Proj', ascending=False).head(15),
                 use_container_width=True)

# Optimizer
st.header("Lineup Optimizer")
if salaries is not None and 'Driver' in salaries.columns and 'Salary' in salaries.columns and 'Final_Proj' in salaries.columns:
    num_lineups = st.slider("Lineups (1 = best only)", 1, 50, 10)
    if st.button("Generate Lineups"):
        with st.spinner("Optimizing..."):
            lineups = []
            base_proj = salaries['Final_Proj'].copy()
            for i in range(num_lineups):
                proj_col = 'Final_Proj' if i == 0 else 'Perturbed'
                if i > 0:
                    salaries[proj_col] = base_proj * np.random.normal(1.0, 0.10, len(salaries))
                else:
                    salaries[proj_col] = base_proj

                prob = pulp.LpProblem("DFS", pulp.LpMaximize)
                select = {d: pulp.LpVariable(f"s_{d}", 0, 1, pulp.LpBinary) for d in salaries['Driver']}

                prob += pulp.lpSum(select[d] * salaries[salaries['Driver'] == d][proj_col].iloc[0] for d in select)
                prob += pulp.lpSum(select.values()) == 6
                prob += pulp.lpSum(select[d] * salaries[salaries['Driver'] == d]['Salary'].iloc[0] for d in select) <= salary_cap

                try:
                    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=8))
                    if prob.status == 1:
                        sel = [d for d in select if select[d].value() > 0.5]
                        pts = sum(salaries[salaries['Driver'] == d][proj_col].iloc[0] for d in sel)
                        sal = sum(salaries[salaries['Driver'] == d]['Salary'].iloc[0] for d in sel)
                        lineups.append({'Lineup': ', '.join(sel), 'Proj': round(pts,1), 'Salary': int(sal)})
                except Exception as e:
                    st.warning(f"Optimizer error on variant {i+1}: {str(e)}")

            if lineups:
                df = pd.DataFrame(lineups).sort_values('Proj', ascending=False)
                st.dataframe(df, use_container_width=True)
                st.download_button("Download CSV", df.to_csv(index=False), "lineups.csv")
            else:
                st.error("No lineups generated – check salaries/projections or try fewer variants.")
else:
    st.info("Upload a salary CSV to enable optimizer.")

with st.expander("Debug Info"):
    st.write(f"Schedule rows: {len(schedule)}")
    st.write(f"History rows: {len(hist)}")
    st.write(f"Salaries loaded: {'Yes' if salaries is not None else 'No'}")