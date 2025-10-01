import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


@st.cache_resource
def load_model():
    model = joblib.load("random_forest_matchup_model.joblib")
    return model



# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("NBA_player_stats.csv")  # make sure you export dataset to CSV
    return df



# Aggregate player stats into team features
def build_team_features(player_names, df):
    team_df = df[df["Player"].isin(player_names)]
    # Aggregate stats (mean or sum depending on feature)
    agg_features = team_df.mean(numeric_only=True)
    return agg_features

# Simulate matchup
def simulate_matchup(teamA, teamB, df, model):
    teamA_features = build_team_features(teamA, df)
    teamB_features = build_team_features(teamB, df)

    # Combine features into one row
    matchup_features = teamA_features.values - teamB_features.values
    matchup_features = matchup_features.reshape(1, -1)

    # Predict probability
    prob = model.predict_proba(matchup_features)[0]
    return prob

# --- Streamlit UI ---
st.title("🏀 NBA Dream Team Matchup Simulator")

df = load_data()
model = load_model()

players = df["Player"].unique()

st.sidebar.header("Select Players")

teamA = st.sidebar.multiselect("Choose 5 players for Team A", players)
teamB = st.sidebar.multiselect("Choose 5 players for Team B", players)

if len(teamA) == 5 and len(teamB) == 5:
    prob = simulate_matchup(teamA, teamB, df, model)

    st.subheader("Winning Probability")
    st.write(f"**Team A ({', '.join(teamA)})**: {prob[1]*100:.2f}%")
    st.write(f"**Team B ({', '.join(teamB)})**: {prob[0]*100:.2f}%")

    # Bar chart
    st.bar_chart({"Team A": prob[1], "Team B": prob[0]})
else:
    st.info("Select 5 players for each team to simulate a matchup.")
