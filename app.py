import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

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
        background-color: #1e3a8a;
        color: white !important;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .player-tag:hover {
        background-color: #2d4a9a;
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }
    .stProgress > div > div > div > div {
        background-color: #1e3a8a;
    }
    /* Fix for multiselect text color */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #1e3a8a;
        color: white;
    }
    .stMultiSelect [data-baseweb="tag"] span {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    data = joblib.load("random_forest_matchup_model.joblib")
    return data["model"], data["features"]

model, feature_names = load_model()


@st.cache_data
def load_data():
    df = pd.read_csv("NBA_Player_Stats.csv")
    df.columns = df.columns.str.strip()
    return df

def build_team_features(player_names, df):
    team_df = df[df["Name"].isin(player_names)]
    agg_features = team_df.mean(numeric_only=True)
    return agg_features

def simulate_matchup(teamA, teamB, df, model, feature_names):
    teamA_features = build_team_features(teamA, df)
    teamB_features = build_team_features(teamB, df)

    if "PTS" in teamA_features.index:
        teamA_features = teamA_features.drop("PTS")
        teamB_features = teamB_features.drop("PTS")

    matchup_features = (teamA_features.values - teamB_features.values).reshape(1, -1)

    # Convert to DataFrame with correct feature names
    matchup_df = pd.DataFrame(matchup_features, columns=feature_names)

    prob = model.predict_proba(matchup_df)[0]
    return prob


def get_player_stats(player_names, df):
    """Get key stats for selected players"""
    # Check if REB column exists, if not try TRB (Total Rebounds)
    stat_columns = ["Name", "PTS", "AST", "STL", "BLK"]
    
    # Add rebounds column (try different common names)
    if "REB" in df.columns:
        stat_columns.append("REB")
    elif "TRB" in df.columns:
        stat_columns.append("TRB")
    elif "Rebounds" in df.columns:
        stat_columns.append("Rebounds")
    
    stats = df[df["Name"].isin(player_names)][stat_columns].set_index("Name")
    return stats

# --- Improved Streamlit UI ---
st.markdown('<h1 class="main-header">🏀 NBA Dream Team Matchup Simulator</h1>', unsafe_allow_html=True)

# Load data
df = load_data()
model = load_model()
players = sorted(df["Name"].unique())  # Sort players alphabetically

# Sidebar with improved organization
with st.sidebar:
    st.header("🏀 Team Builder")
    
    st.subheader("Team A")
    teamA = st.multiselect(
        "Select 5 players for Team A", 
        players, 
        key="teamA",
        help="Choose 5 players to form Team A",
        max_selections=5
    )
    
    st.subheader("Team B") 
    teamB = st.multiselect(
        "Select 5 players for Team B", 
        players, 
        key="teamB",
        help="Choose 5 players to form Team B",
        max_selections=5
    )
    
    # Team validation with better messaging
    if len(teamA) > 5:
        st.warning("⚠️ Team A can only have 5 players. Please remove extras.")
    
    if len(teamB) > 5:
        st.warning("⚠️ Team B can only have 5 players. Please remove extras.")
    
    # Show current team counts
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Team A Players", f"{len(teamA)}/5")
    with col2:
        st.metric("Team B Players", f"{len(teamB)}/5")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    st.subheader("👑 Team A")
    if teamA:
        st.write("**Selected Players:**")
        for player in teamA:
            st.markdown(f'<span class="player-tag">{player}</span>', unsafe_allow_html=True)
    else:
        st.info("Select players from the sidebar to build Team A")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    st.subheader("⚡ Team B")
    if teamB:
        st.write("**Selected Players:**")
        for player in teamB:
            st.markdown(f'<span class="player-tag">{player}</span>', unsafe_allow_html=True)
    else:
        st.info("Select players from the sidebar to build Team B")
    st.markdown('</div>', unsafe_allow_html=True)

# Simulation and Results
if len(teamA) == 5 and len(teamB) == 5:
    st.markdown("---")
    
    # Add a simulate button for better UX
    if st.button("🚀 Simulate Matchup", use_container_width=True, type="primary"):
        with st.spinner("🏀 Simulating matchup... Analyzing player stats and predicting outcome..."):
            prob = simulate_matchup(teamA, teamB, df, model)
            
            # Results in columns
            st.subheader("🎯 Matchup Results")
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.markdown('<div class="probability-card">', unsafe_allow_html=True)
                st.metric(
                    label="Team A Win Probability", 
                    value=f"{prob[1]*100:.1f}%",
                    delta=f"+{prob[1]*100 - 50:.1f}%" if prob[1] > 0.5 else f"{prob[1]*100 - 50:.1f}%"
                )
                st.progress(prob[1])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("<h2 style='text-align: center; color: #666;'>VS</h2>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="probability-card">', unsafe_allow_html=True)
                st.metric(
                    label="Team B Win Probability", 
                    value=f"{prob[0]*100:.1f}%",
                    delta=f"+{prob[0]*100 - 50:.1f}%" if prob[0] > 0.5 else f"{prob[0]*100 - 50:.1f}%"
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
                colors = ['#FF6B6B', '#4ECDC4']
                ax.pie([prob[1], prob[0]], 
                      labels=['Team A', 'Team B'], 
                      autopct='%1.1f%%',
                      colors=colors,
                      startangle=90,
                      textprops={'fontsize': 12})
                ax.set_title('Win Probability Distribution', fontsize=14, fontweight='bold')
                st.pyplot(fig)
            
            # Team stats comparison
            st.subheader("📈 Team Stats Comparison")
            try:
                teamA_stats = get_player_stats(teamA, df)
                teamB_stats = get_player_stats(teamB, df)
                
                avgA = teamA_stats.mean()
                avgB = teamB_stats.mean()
                
                # Create readable stat names
                stat_names = []
                for col in teamA_stats.columns:
                    if col == 'PTS': stat_names.append('Points')
                    elif col == 'AST': stat_names.append('Assists')
                    elif col in ['REB', 'TRB', 'Rebounds']: stat_names.append('Rebounds')
                    elif col == 'STL': stat_names.append('Steals')
                    elif col == 'BLK': stat_names.append('Blocks')
                    else: stat_names.append(col)
                
                stats_comparison = pd.DataFrame({
                    'Stat': stat_names,
                    'Team A': [f"{val:.1f}" for val in avgA.values],
                    'Team B': [f"{val:.1f}" for val in avgB.values]
                })
                
                st.dataframe(stats_comparison, use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Could not load detailed stats: {e}")
            
    else:
        st.info("👆 Click the button above to simulate the matchup!")
        
else:
    st.markdown("---")
    # More helpful messaging when teams aren't complete
    if len(teamA) != 5 or len(teamB) != 5:
        st.info("🎯 **Build Your Teams**: Select 5 players for each team to unlock matchup simulation")
        
        col1, col2 = st.columns(2)
        with col1:
            if len(teamA) < 5:
                st.warning(f"Team A needs {5 - len(teamA)} more player(s)")
            elif len(teamA) == 5:
                st.success("✅ Team A is ready!")
                
        with col2:
            if len(teamB) < 5:
                st.warning(f"Team B needs {5 - len(teamB)} more player(s)")
            elif len(teamB) == 5:
                st.success("✅ Team B is ready!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.9rem;'>"
    "Built with Streamlit • NBA Dream Team Simulator"
    "</div>",
    unsafe_allow_html=True
)

