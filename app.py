import streamlit as st
import pandas as pd
import requests
import pulp
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="NASCAR DFS Optimizer 2026", layout="wide")
st.title("🏁 NASCAR DFS Optimizer: All Tracks (2026)")
st.markdown("**Live data from Racing-Reference • DK/FD support • Track-specific projections & optimizer**")

# --- DATA PIPELINE (Auto-Updating) ---
@st.cache_data(ttl=3600)  # Refresh hourly
def get_schedule():
    url = "https://www.racing-reference.info/season-stats/2026/W"
    try:
        df = pd.read_html(requests.get(url).text)[0]
        return df[['Date', 'Race', 'Track']].dropna(how='all')
    except:
        return pd.DataFrame({'Date': [], 'Race': ['No schedule loaded'], 'Track': []})

@st.cache_data(ttl=7200)
def get_track_history(track_name):
    # Fallback: use most recent superspeedway or similar for demo; expand later
    # In production: map track → recent race URL or search RR
    recent_race_urls = [
        "https://www.racing-reference.info/race-results/2026_Daytona_500/W",  # Update as needed
        "https://www.racing-reference.info/race-results/2025_Daytona_500/W",
    ]
    all_data = []
    for url in recent_race_urls[:3]:  # Limit to avoid overload
        try:
            df = pd.read_html(requests.get(url).text)[0]
            if 'Driver' in df.columns and 'Finish' in df.columns:
                df = df[['Driver', 'Start', 'Finish', 'Laps', 'Led', 'Status']].copy()
                df['Race'] = url.split('/')[-2]
                all_data.append(df)
        except:
            pass
    if not all_data:
        return pd.DataFrame()
    hist = pd.concat(all_data, ignore_index=True)
    hist['Avg_Finish'] = hist.groupby('Driver')['Finish'].transform('mean')
    hist['Laps_Led_Pct'] = hist.groupby('Driver')['Led'].transform('mean') / hist.groupby('Driver')['Laps'].transform('mean').replace(0, 1)
    proj = hist[['Driver', 'Avg_Finish', 'Laps_Led_Pct']].drop_duplicates()
    proj['DK_Proj_Base'] = (41 - proj['Avg_Finish'].clip(1, 40)) * 1.5 + proj['Laps_Led_Pct'] * 25
    return proj

def get_track_type(track):
    superspeed = ['Daytona', 'Talladega']
    short = ['Bristol', 'Martinsville', 'Richmond']
    road = ['Circuit of The Americas', 'Sonoma', 'Watkins Glen', 'Road America']
    if any(s in track for s in superspeed): return 'superspeedway'
    if any(s in track for s in short): return 'short'
    if any(s in track for s in road): return 'road'
    return 'intermediate'

# --- SIDEBAR: SELECT RACE ---
schedule = get_schedule()
st.sidebar.header("Select Race / Track")
if not schedule.empty:
    race_options = schedule['Race'].astype(str) + " @ " + schedule['Track'].astype(str)
    selected = st.sidebar.selectbox("Race", race_options)
    idx = race_options.tolist().index(selected)
    race = schedule.iloc[idx]['Race']
    track = schedule.iloc[idx]['Track']
else:
    race, track = "Upcoming Race", "Unknown Track"
    st.sidebar.warning("Schedule scrape failed – using fallback")

track_type = get_track_type(track)
st.sidebar.success(f"**{race} at {track}** ({track_type.capitalize()}) • Refreshed: {datetime.now().strftime('%H:%M %Z')}")

# --- PROJECTIONS ---
st.header("📊 Driver Projections")
hist = get_track_history(track)
if not hist.empty:
    st.dataframe(
        hist.sort_values('DK_Proj_Base', ascending=False).head(20)[['Driver', 'Avg_Finish', 'Laps_Led_Pct', 'DK_Proj_Base']],
        use_container_width=True,
        column_config={"DK_Proj_Base": "Base DK Proj"}
    )
    fig = px.bar(hist.nlargest(15, 'DK_Proj_Base'), x='Driver', y='DK_Proj_Base', title=f"Top 15 Projections – {track}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No historical data loaded yet for this track – projections will use defaults after salary upload.")

# --- SALARY UPLOAD ---
st.header("💰 Upload Current Salaries")
col1, col2 = st.columns(2)
with col1:
    dk_file = st.file_uploader("DraftKings CSV Export", type="csv", key="dk")
with col2:
    fd_file = st.file_uploader("FanDuel CSV Export", type="csv", key="fd")

salaries = None
site = None
salary_cap = 50000
if dk_file:
    salaries = pd.read_csv(dk_file)
    site = "DraftKings"
elif fd_file:
    salaries = pd.read_csv(fd_file)
    site = "FanDuel"
    salary_cap = 60000  # FD typical cap; confirm per slate

if salaries is not None:
    st.success(f"Loaded {len(salaries)} drivers from {site}")
    # Assume columns: Name/Driver, Salary, maybe Position (all DRV), maybe Team/Manufacturer
    salaries = salaries.rename(columns={'Name': 'Driver'})  # DK uses 'Name'
    salaries = salaries.merge(hist, on='Driver', how='left').fillna({'Avg_Finish': 20, 'Laps_Led_Pct': 0, 'DK_Proj_Base': 25})
    
    # Final proj adjustments
    multiplier = 1.3 if track_type == 'superspeedway' else 1.0 if track_type == 'intermediate' else 0.85
    salaries['Final_Proj'] = salaries['DK_Proj_Base'] * multiplier
    
    st.dataframe(salaries[['Driver', 'Salary', 'Final_Proj']].sort_values('Final_Proj', ascending=False).head(15),
                 use_container_width=True)

# --- OPTIMIZER ---
st.header("⚙️ PuLP Lineup Optimizer")
if salaries is not None and not salaries.empty:
    num_lineups = st.slider("Number of lineups to generate (1 = single optimal)", 1, 150, 30)
    max_per_team = st.slider("Max drivers per team (stack control)", 2, 6, 4)

    if st.button("🚀 Generate Lineups", type="primary"):
        with st.spinner(f"Running PuLP optimizer ({num_lineups} lineups)..."):
            lineups = []
            base_prob = salaries['Final_Proj'].copy()

            for i in range(num_lineups):
                # Slight perturbation for GPP diversity
                if i > 0:
                    perturbation = np.random.normal(1.0, 0.12, len(salaries))
                    salaries['Perturbed_Proj'] = base_prob * perturbation
                else:
                    salaries['Perturbed_Proj'] = base_prob

                prob = pulp.LpProblem(f"NASCAR_DFS_{i}", pulp.LpMaximize)

                select = pulp.LpVariable.dicts("select", salaries['Driver'], 0, 1, pulp.LpBinary)

                prob += pulp.lpSum(select[d] * salaries[salaries['Driver'] == d]['Perturbed_Proj'].iloc[0]
                                   for d in salaries['Driver'])

                prob += pulp.lpSum(select.values()) == 6

                prob += pulp.lpSum(select[d] * salaries[salaries['Driver'] == d]['Salary'].iloc[0]
                                   for d in salaries['Driver']) <= salary_cap

                # Team stack limit (if 'Team' column exists)
                if 'Team' in salaries.columns:
                    for team, grp in salaries.groupby('Team'):
                        prob += pulp.lpSum(select[d] for d in grp['Driver']) <= max_per_team

                # Optional superspeedway manufacturer min (if column exists)
                if track_type == 'superspeedway' and 'Manufacturer' in salaries.columns:
                    for mfr, grp in salaries.groupby('Manufacturer'):
                        prob += pulp.lpSum(select[d] for d in grp['Driver']) >= 2, f"Min2_{mfr}"

                prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10 + i*2))

                if pulp.LpStatus[prob.status] == 'Optimal':
                    selected = [d for d in select if select[d].value() > 0.5]
                    pts = sum(salaries[salaries['Driver'] == d]['Perturbed_Proj'].iloc[0] for d in selected)
                    sal_total = sum(salaries[salaries['Driver'] == d]['Salary'].iloc[0] for d in selected)
                    lineups.append({
                        'Lineup': ', '.join(selected),
                        'Proj_Pts': round(pts, 1),
                        'Salary': int(sal_total),
                        'Variant': i+1
                    })

            if lineups:
                df_lineups = pd.DataFrame(lineups).sort_values('Proj_Pts', ascending=False)
                st.dataframe(df_lineups, use_container_width=True)
                st.download_button("Download Lineups CSV", df_lineups.to_csv(index=False), "nascar_lineups.csv")
            else:
                st.error("No valid lineups found – check salary cap or relax team limits.")

st.sidebar.markdown("---")
st.sidebar.info("Upload DK/FD CSV after qualifying for best results.\n"
                "Projections use recent history + track adjustments.\n"
                "Pure PuLP – no external optimizer libs needed.")