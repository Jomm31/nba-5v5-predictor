import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from streamlit_extras.card import card

# Page configuration
st.set_page_config(
    page_title="NBA Dream Team Simulator",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .team-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #1e3a8a;
        margin-bottom: 1rem;
    }
    .probability-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .player-tag {
        background-color: #e2e8f0;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1e3a8a;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = joblib.load("random_forest_matchup_model.joblib")
        return model
    except FileNotFoundError:
        st.error("❌ Model file `random_forest_matchup_model.joblib` not found. Please add it to the repo root.")
        st.stop()

@st.cache_data
def load_data():
    df = pd.read_csv("NBA_Player_Stats.csv")
    df.columns = df.columns.str.strip()
    return df

def build_team_features(player_names, df):
    team_df = df[df["Name"].isin(player_names)]
    agg_features = team_df.mean(numeric_only=True)
    return agg_features

def simulate_matchup(teamA, teamB, df, model):
    teamA_features = build_team_features(teamA, df)
    teamB_features = build_team_features(teamB, df)

    if "PTS" in teamA_features.index:
        teamA_features = teamA_features.drop("PTS")
        teamB_features = teamB_features.drop("PTS")

    matchup_features = teamA_features.values - teamB_features.values
    matchup_features = matchup_features.reshape(1, -1)

    prob = model.predict_proba(matchup_features)[0]
    return prob

def get_player_stats(player_names, df):
    """Get key stats for selected players"""
    stats = df[df["Name"].isin(player_names)][["Name", "PTS", "AST", "TRB", "STL", "BLK"]].set_index("Name")
    return stats

# --- Improved Streamlit UI ---
st.markdown('<h1 class="main-header">🏀 NBA Dream Team Matchup Simulator</h1>', unsafe_allow_html=True)

# Load data
df = load_data()
model = load_model()
players = df["Name"].unique()

# Sidebar with improved organization
with st.sidebar:
    st.header("🏀 Team Builder")
    
    st.subheader("Team A")
    teamA = st.multiselect(
        "Select 5 players for Team A", 
        players, 
        key="teamA",
        help="Choose 5 players to form Team A"
    )
    
    st.subheader("Team B") 
    teamB = st.multiselect(
        "Select 5 players for Team B", 
        players, 
        key="teamB",
        help="Choose 5 players to form Team B"
    )
    
    # Team validation
    if len(teamA) > 5:
        st.warning("⚠️ Team A can only have 5 players. Removing extras.")
        teamA = teamA[:5]
    
    if len(teamB) > 5:
        st.warning("⚠️ Team B can only have 5 players. Removing extras.")
        teamB = teamB[:5]

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    st.subheader("👑 Team A")
    if teamA:
        for player in teamA:
            st.markdown(f'<span class="player-tag">{player}</span>', unsafe_allow_html=True)
    else:
        st.info("Select players from the sidebar")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    st.subheader("⚡ Team B")
    if teamB:
        for player in teamB:
            st.markdown(f'<span class="player-tag">{player}</span>', unsafe_allow_html=True)
    else:
        st.info("Select players from the sidebar")
    st.markdown('</div>', unsafe_allow_html=True)

# Simulation and Results
if len(teamA) == 5 and len(teamB) == 5:
    st.markdown("---")
    
    # Add a simulate button for better UX
    if st.button("🚀 Simulate Matchup", use_container_width=True):
        with st.spinner("Simulating matchup... This may take a few seconds."):
            prob = simulate_matchup(teamA, teamB, df, model)
            
            # Results in columns
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.markdown('<div class="probability-card">', unsafe_allow_html=True)
                st.metric(
                    label="Team A Win Probability", 
                    value=f"{prob[1]*100:.1f}%",
                    delta=f"+{prob[1]*100 - 50:.1f}%" if prob[1] > 0.5 else None
                )
                st.progress(prob[1])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.subheader("VS")
                st.markdown("<br>", unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="probability-card">', unsafe_allow_html=True)
                st.metric(
                    label="Team B Win Probability", 
                    value=f"{prob[0]*100:.1f}%",
                    delta=f"+{prob[0]*100 - 50:.1f}%" if prob[0] > 0.5 else None
                )
                st.progress(prob[0])
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualization section
            st.subheader("📊 Matchup Analysis")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Bar chart
                chart_data = pd.DataFrame({
                    'Team': ['Team A', 'Team B'],
                    'Win Probability': [prob[1], prob[0]]
                })
                st.bar_chart(chart_data.set_index('Team'), use_container_width=True)
            
            with viz_col2:
                # Pie chart
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie([prob[1], prob[0]], 
                      labels=['Team A', 'Team B'], 
                      autopct='%1.1f%%',
                      colors=['#FF6B6B', '#4ECDC4'],
                      startangle=90)
                ax.set_title('Win Probability Distribution')
                st.pyplot(fig)
            
            # Team stats comparison
            st.subheader("📈 Team Stats Comparison")
            teamA_stats = get_player_stats(teamA, df)
            teamB_stats = get_player_stats(teamB, df)
            
            avgA = teamA_stats.mean()
            avgB = teamB_stats.mean()
            
            stats_comparison = pd.DataFrame({
                'Stat': ['Points', 'Assists', 'Rebounds', 'Steals', 'Blocks'],
                'Team A': avgA.values,
                'Team B': avgB.values
            })
            
            st.dataframe(stats_comparison, use_container_width=True, hide_index=True)
            
    else:
        st.info("👆 Click the button above to simulate the matchup!")
        
else:
    st.markdown("---")
    if len(teamA) != 5 or len(teamB) != 5:
        col1, col2 = st.columns(2)
        with col1:
            if len(teamA) != 5:
                st.warning(f"Team A needs {5 - len(teamA)} more player(s)")
        with col2:
            if len(teamB) != 5:
                st.warning(f"Team B needs {5 - len(teamB)} more player(s)")
    
    # Placeholder for when teams aren't complete
    st.info("🎯 Select 5 players for each team to unlock matchup simulation")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with Streamlit • NBA Dream Team Simulator"
    "</div>",
    unsafe_allow_html=True
)
