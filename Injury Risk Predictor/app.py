"""
Injury Risk Predictor - Streamlit Dashboard
AI-powered injury risk assessment for Premier League players
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="Injury Risk Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .risk-high {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #c62828;
    }
    .risk-medium {
        background-color: #fff3e0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ef6c00;
    }
    .risk-low {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2e7d32;
    }
    .flag-red { color: #c62828; font-weight: bold; }
    .flag-amber { color: #ef6c00; font-weight: bold; }
    .flag-green { color: #2e7d32; font-weight: bold; }
    .player-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .stat-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .high-risk-badge {
        background-color: #c62828;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 12px;
    }
    .medium-risk-badge {
        background-color: #ef6c00;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 12px;
    }
    .low-risk-badge {
        background-color: #2e7d32;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 12px;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# DATA & MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models():
    """Load trained model artifacts. Returns dict or None (demo mode)."""
    try:
        from src.utils.model_io import load_artifacts
        artifacts = load_artifacts()
        return artifacts
    except Exception:
        return None


def get_data_freshness():
    """Get timestamp of when the inference data was last updated."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    inference_path = os.path.join(models_dir, "inference_df.pkl")

    if os.path.exists(inference_path):
        mtime = os.path.getmtime(inference_path)
        dt = datetime.fromtimestamp(mtime)
        now = datetime.now()
        delta = now - dt

        if delta.days == 0:
            if delta.seconds < 3600:
                human = f"{delta.seconds // 60} minutes ago"
            else:
                human = f"{delta.seconds // 3600} hours ago"
        elif delta.days == 1:
            human = "yesterday"
        elif delta.days < 7:
            human = f"{delta.days} days ago"
        else:
            human = dt.strftime("%b %d, %Y")

        return dt, human
    return None, None


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def generate_real_prediction(inference_df, player_name):
    """Generate prediction from pre-computed inference DataFrame."""
    from src.dashboard.player_dashboard import get_latest_snapshot
    from src.inference.risk_card import build_risk_card

    try:
        row = get_latest_snapshot(inference_df, player_name)
    except (KeyError, IndexError):
        return None

    card = build_risk_card(row)

    risk_level = card["risk_assessment"]["level"].title()
    risk_prob = card["risk_assessment"]["probability"]

    if risk_level == "Critical":
        risk_color = "high"
    elif risk_level == "High":
        risk_color = "high"
    elif risk_level == "Moderate":
        risk_color = "medium"
    else:
        risk_color = "low"

    support = card["risk_assessment"]["model_support"]
    ensemble_probs = {
        "catboost": support.get("catboost", risk_prob),
        "lightgbm": support.get("lightgbm", risk_prob),
        "xgboost": support.get("xgboost", risk_prob),
    }

    top_factors = []
    for f in card.get("top_factors", []):
        impact = f["impact"]
        if f["direction"] == "decrease_risk" and impact > 0:
            impact = -impact
        top_factors.append((f["feature"], impact))

    return {
        "risk_probability": risk_prob,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "severity_days": card["severity_projection"]["projected_days_lost"],
        "severity_level": card["severity_projection"]["severity_level"].title(),
        "archetype": card["archetype"]["name"],
        "archetype_profile": card["archetype"]["profile"],
        "top_factors": top_factors,
        "confidence": card["risk_assessment"].get("confidence", "medium"),
        "confidence_score": support.get("agreement_score", 2) / 3.0,
        "ensemble_probs": ensemble_probs,
        "training_flag": card.get("training_flag", "Amber"),
        "recommendation": card.get("recommendation", ""),
        "injury_history": card.get("injury_history"),
        "team": card.get("team", "Unknown"),
        "position": card.get("position", "Unknown"),
    }


# ============================================================================
# DISPLAY COMPONENTS
# ============================================================================

def get_risk_badge(risk_prob):
    """Return HTML badge for risk level."""
    if risk_prob >= 0.6:
        return f'<span class="high-risk-badge">{risk_prob:.0%} High</span>'
    elif risk_prob >= 0.35:
        return f'<span class="medium-risk-badge">{risk_prob:.0%} Medium</span>'
    else:
        return f'<span class="low-risk-badge">{risk_prob:.0%} Low</span>'


def get_archetype_info(archetype):
    """Return archetype description."""
    profiles = {
        "Currently Vulnerable": {
            "emoji": "🔴",
            "desc": "Recently returned from injury - high re-injury risk",
            "advice": "Limit minutes, gradual return to full intensity"
        },
        "Fragile": {
            "emoji": "🟠",
            "desc": "When injuries happen, they tend to be serious",
            "advice": "Careful load management, avoid overexertion"
        },
        "Injury Prone": {
            "emoji": "🟡",
            "desc": "Frequent injuries but usually recovers quickly",
            "advice": "Monitor workload spikes, consistent training"
        },
        "Recurring": {
            "emoji": "🟡",
            "desc": "Frequent minor injuries, often same areas",
            "advice": "Targeted strengthening, prehab exercises"
        },
        "Moderate Risk": {
            "emoji": "🟢",
            "desc": "Average injury profile for position",
            "advice": "Standard training protocols apply"
        },
        "Durable": {
            "emoji": "🟢",
            "desc": "Handles high workloads well, quick recovery",
            "advice": "Can sustain high minutes, normal rotation"
        },
        "Clean Record": {
            "emoji": "⚪",
            "desc": "No significant injury history recorded",
            "advice": "Monitor as normal, build baseline data"
        },
    }
    return profiles.get(archetype, {"emoji": "⚪", "desc": "Unknown profile", "advice": ""})


def display_player_card(prediction, player_name, row):
    """Display the main player risk card."""
    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])

    with col1:
        st.markdown(f"## {player_name}")
        team = prediction.get('team', row.get('team', 'Unknown'))
        st.markdown(f"**{team}**")

    with col2:
        risk_prob = prediction['risk_probability']
        risk_level = prediction['risk_level']
        delta_color = "inverse" if risk_prob < 0.35 else ("off" if risk_prob < 0.6 else "normal")
        st.metric("Injury Risk", f"{risk_prob:.0%}", risk_level, delta_color=delta_color)

    with col3:
        st.metric("Archetype", prediction['archetype'])

    with col4:
        prev_inj = row.get('previous_injuries', 0)
        days_since = row.get('days_since_last_injury', 'N/A')
        if isinstance(days_since, (int, float)):
            days_since = int(days_since)
        st.metric("Days Since Injury", days_since, f"{int(prev_inj)} total injuries")


def display_risk_details(prediction):
    """Display detailed risk analysis."""
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Risk Analysis")

        # Risk gauge visualization
        risk_prob = prediction['risk_probability']
        risk_color = "#c62828" if risk_prob >= 0.6 else ("#ef6c00" if risk_prob >= 0.35 else "#2e7d32")

        st.markdown(f"""
        <div class="risk-{prediction['risk_color']}" style="margin-bottom: 20px;">
            <h3 style="margin: 0;">{prediction['risk_level']} Risk</h3>
            <p style="font-size: 2em; margin: 10px 0; font-weight: bold;">{risk_prob:.0%}</p>
            <p>Training Flag: <strong>{prediction.get('training_flag', 'N/A')}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Model agreement
        st.markdown("#### Model Predictions")
        ensemble = prediction['ensemble_probs']
        model_cols = st.columns(3)
        with model_cols[0]:
            st.metric("CatBoost", f"{ensemble['catboost']:.0%}")
        with model_cols[1]:
            st.metric("LightGBM", f"{ensemble['lightgbm']:.0%}")
        with model_cols[2]:
            st.metric("XGBoost", f"{ensemble['xgboost']:.0%}")

    with col_right:
        st.markdown("### Archetype Profile")

        archetype = prediction['archetype']
        arch_info = get_archetype_info(archetype)

        st.markdown(f"""
        <div class="stat-box">
            <h2>{arch_info['emoji']} {archetype}</h2>
            <p>{arch_info['desc']}</p>
            <p><strong>Advice:</strong> {arch_info['advice']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Recommendations
        st.markdown("#### Recommendations")
        risk = prediction['risk_level']
        if risk in ("High", "Critical"):
            st.warning("⚠️ Reduce training intensity by 20-30%")
            st.warning("⏱️ Limit match minutes to 60-70 where possible")
        elif risk == "Moderate":
            st.info("⚡ Monitor workload - avoid sudden increases")
            st.info("⏱️ Manage minutes to ~70-80, avoid extra time")
        else:
            st.success("✅ Continue current training structure")
            st.success("⏱️ Full availability under normal rotation")


def display_team_overview(inference_df, selected_team):
    """Display team overview with all players sorted by risk."""
    if selected_team == "All Teams":
        team_df = inference_df.copy()
        st.markdown("### All EPL Players by Risk")
    else:
        team_df = inference_df[inference_df['team'] == selected_team].copy()
        st.markdown(f"### {selected_team} Squad")

    # Sort by risk (highest first)
    team_df = team_df.sort_values('ensemble_prob', ascending=False)

    # Summary stats
    high_risk = len(team_df[team_df['ensemble_prob'] >= 0.6])
    medium_risk = len(team_df[(team_df['ensemble_prob'] >= 0.35) & (team_df['ensemble_prob'] < 0.6)])
    low_risk = len(team_df[team_df['ensemble_prob'] < 0.35])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", len(team_df))
    with col2:
        st.metric("High Risk", high_risk, delta=None)
    with col3:
        st.metric("Medium Risk", medium_risk)
    with col4:
        st.metric("Low Risk", low_risk)

    st.markdown("---")

    # Display players in a table with risk badges
    for idx, (_, row) in enumerate(team_df.iterrows()):
        if idx >= 50:  # Limit display
            st.caption(f"... and {len(team_df) - 50} more players")
            break

        risk_prob = row['ensemble_prob']
        if risk_prob >= 0.6:
            risk_emoji = "🔴"
            risk_text = "High"
        elif risk_prob >= 0.35:
            risk_emoji = "🟠"
            risk_text = "Medium"
        else:
            risk_emoji = "🟢"
            risk_text = "Low"

        days_since = row.get('days_since_last_injury', 'N/A')
        if isinstance(days_since, (int, float)):
            days_since = f"{int(days_since)}d"

        prev_inj = int(row.get('previous_injuries', 0))
        archetype = row.get('archetype', 'Unknown')
        arch_info = get_archetype_info(archetype)

        col1, col2, col3, col4, col5 = st.columns([3, 1, 1.5, 1.5, 2])
        with col1:
            st.markdown(f"**{row['name']}**")
        with col2:
            st.markdown(f"{risk_emoji} {risk_prob:.0%}")
        with col3:
            st.caption(f"Since injury: {days_since}")
        with col4:
            st.caption(f"Injuries: {prev_inj}")
        with col5:
            st.caption(f"{arch_info['emoji']} {archetype}")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load models
    artifacts = load_models()
    production_mode = artifacts is not None

    if not production_mode:
        st.title("⚽ Injury Risk Predictor")
        st.error("No trained models found. Run the training notebook first.")
        return

    inference_df = artifacts["inference_df"]
    data_dt, data_human = get_data_freshness()

    # Header
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.title("⚽ EPL Injury Risk Predictor")
    with header_col2:
        st.caption(f"Data: {inference_df['name'].nunique()} players")
        if data_human:
            st.caption(f"Updated: {data_human}")

    # How it works - compact explanation
    with st.expander("ℹ️ How This Works", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **1. Injury History Analysis**
            - Previous injuries from Transfermarkt
            - Days since last injury
            - Average severity (days out)
            """)
        with col2:
            st.markdown("""
            **2. Risk Factors**
            - Recency of last injury
            - Injury severity patterns
            - Career injury frequency
            """)
        with col3:
            st.markdown("""
            **3. Player Archetypes**
            - Durable, Fragile, Injury Prone
            - Currently Vulnerable (just returned)
            - Clean Record (no history)
            """)

    # Sidebar
    st.sidebar.header("⚽ Select Player")

    all_players = inference_df[["name", "team", "archetype", "ensemble_prob"]].drop_duplicates("name")
    all_teams = sorted(all_players["team"].unique().tolist())

    # Team filter
    selected_team = st.sidebar.selectbox(
        "Team",
        options=["All Teams"] + all_teams,
        index=0
    )

    # Filter players
    if selected_team == "All Teams":
        filtered_players = all_players.sort_values("ensemble_prob", ascending=False)
    else:
        filtered_players = all_players[all_players["team"] == selected_team].sort_values("ensemble_prob", ascending=False)

    # Player dropdown with risk indicator
    def format_player(row):
        risk = row['ensemble_prob']
        if risk >= 0.6:
            emoji = "🔴"
        elif risk >= 0.35:
            emoji = "🟠"
        else:
            emoji = "🟢"
        return f"{emoji} {row['name']} ({risk:.0%})"

    player_options = filtered_players.apply(format_player, axis=1).tolist()
    player_name_lookup = dict(zip(player_options, filtered_players["name"].tolist()))

    selected_option = st.sidebar.selectbox(
        "Player",
        options=[""] + player_options,
        index=0,
        help=f"{len(filtered_players)} players"
    )

    player_name = player_name_lookup.get(selected_option, "")

    # View mode
    view_mode = st.sidebar.radio(
        "View",
        ["Player Analysis", "Team Overview"],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with CatBoost, LightGBM, XGBoost")
    st.sidebar.caption("Data: Transfermarkt injury records")

    # Main content
    if view_mode == "Team Overview":
        display_team_overview(inference_df, selected_team)

    elif player_name:
        # Get prediction
        from src.dashboard.player_dashboard import get_latest_snapshot

        try:
            row = get_latest_snapshot(inference_df, player_name)
        except Exception:
            st.error(f"Player '{player_name}' not found")
            return

        prediction = generate_real_prediction(inference_df, player_name)

        if prediction is None:
            st.error(f"Could not generate prediction for {player_name}")
            return

        # Display player card
        display_player_card(prediction, player_name, row)

        st.markdown("---")

        # Display risk details
        display_risk_details(prediction)

    else:
        # Welcome state
        st.markdown("---")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            ### Select a player to analyze

            Use the sidebar to:
            1. **Filter by team** - narrow down to a specific squad
            2. **Select a player** - sorted by risk level (highest first)
            3. **View Team Overview** - see all players at a glance

            Risk levels are color-coded:
            - 🔴 **High Risk** (60%+) - Recently injured or fragile
            - 🟠 **Medium Risk** (35-60%) - Moderate concern
            - 🟢 **Low Risk** (<35%) - Durable or clean record
            """)

        with col2:
            # Quick stats
            high_risk = len(inference_df[inference_df['ensemble_prob'] >= 0.6])
            vulnerable = len(inference_df[inference_df['archetype'] == 'Currently Vulnerable'])

            st.markdown("#### Quick Stats")
            st.metric("High Risk Players", high_risk, f"of {len(inference_df)} total")
            st.metric("Currently Vulnerable", vulnerable, "just returned from injury")

    # Footer
    st.markdown("---")
    st.caption("⚽ EPL Injury Risk Predictor | For educational purposes only | Not medical advice")


if __name__ == "__main__":
    main()
