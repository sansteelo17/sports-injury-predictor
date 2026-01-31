"""
Injury Risk Predictor - Streamlit Dashboard
A proof-of-concept interface for predicting football player injury risk
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

# Custom CSS for better styling
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
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
    .disclaimer {
        background-color: #fff9c4;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #f57f17;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load models with caching
@st.cache_resource
def load_models():
    """
    Attempt to load trained models. Returns None if models not available.
    """
    try:
        # Check if models exist
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        if not os.path.exists(models_dir):
            return None

        # TODO: Add actual model loading logic here when models are trained
        # For now, return None to use mock predictions
        return None
    except Exception as e:
        st.warning(f"Could not load models: {e}")
        return None

@st.cache_data
def load_player_data():
    """
    Load player data for dropdowns and reference.
    """
    try:
        # Load the injuries dataset for teams and positions
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'player_injuries_impact.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            teams = sorted(df['Team Name'].unique().tolist())
            positions = sorted(df['Position'].unique().tolist())
            return teams, positions
        else:
            # Fallback defaults
            return get_default_teams_positions()
    except Exception:
        return get_default_teams_positions()

def get_default_teams_positions():
    """Fallback teams and positions."""
    teams = [
        'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
        'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Leeds',
        'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle',
        'Nottingham Forest', 'Southampton', 'Tottenham', 'West Ham', 'Wolves'
    ]
    positions = [
        'Goalkeeper', 'Center Back', 'Right Back', 'Left Back',
        'Defensive Midfielder', 'Central Midfielder', 'Attacking Midfielder',
        'Right Winger', 'Left Winger', 'Center Forward'
    ]
    return teams, positions

def generate_mock_prediction(player_data):
    """
    Generate mock prediction based on player inputs.
    Uses heuristics to create realistic-looking predictions.
    """
    # Seed based on player data for consistency
    np.random.seed(hash(player_data['name']) % 2**32)

    # Risk factors based on inputs
    age_risk = (player_data['age'] - 16) / 24  # 16-40 normalized
    match_risk = min(player_data['recent_matches'] / 15, 1.0)  # High match load = higher risk

    # Position risk multipliers
    position_risk = {
        'Goalkeeper': 0.3,
        'Center Back': 0.6,
        'Right Back': 0.7,
        'Left Back': 0.7,
        'Defensive Midfielder': 0.65,
        'Central Midfielder': 0.7,
        'Attacking Midfielder': 0.75,
        'Right Winger': 0.8,
        'Left Winger': 0.8,
        'Center Forward': 0.85
    }

    pos_risk = position_risk.get(player_data['position'], 0.6)

    # Calculate base risk with some randomness
    base_risk = (age_risk * 0.3 + match_risk * 0.4 + pos_risk * 0.3)
    risk_prob = np.clip(base_risk + np.random.normal(0, 0.1), 0.1, 0.9)

    # Determine risk level
    if risk_prob >= 0.60:
        risk_level = "High"
        risk_color = "high"
    elif risk_prob >= 0.40:
        risk_level = "Moderate"
        risk_color = "medium"
    else:
        risk_level = "Low"
        risk_color = "low"

    # Severity prediction (correlated with risk)
    severity_days = int(risk_prob * 45 + np.random.normal(0, 10))
    severity_days = max(3, severity_days)

    if severity_days >= 30:
        severity_level = "Catastrophic"
    elif severity_days >= 14:
        severity_level = "Major"
    elif severity_days >= 7:
        severity_level = "Moderate"
    else:
        severity_level = "Minor"

    # Assign archetype based on risk and severity
    if risk_prob >= 0.7 and severity_days >= 30:
        archetype = "Catastrophic + Re-aggravation"
    elif risk_prob >= 0.6 and match_risk > 0.7:
        archetype = "High-Risk Frequent"
    elif risk_prob >= 0.5 and severity_days >= 14:
        archetype = "Moderate-Risk Recurrent"
    elif risk_prob < 0.4:
        archetype = "Low-Severity Stable"
    else:
        archetype = "Moderate-Load High-Variance"

    # Generate top risk factors
    all_factors = [
        ('Recent match load (acute workload)', match_risk * 0.15),
        ('Age-related risk factor', age_risk * 0.12),
        ('Position injury rate', pos_risk * 0.10),
        ('Workload variability', np.random.uniform(0.05, 0.15)),
        ('Previous injury history', np.random.uniform(-0.05, 0.10)),
        ('Recovery time ratio', np.random.uniform(-0.08, 0.05)),
        ('Sprint intensity exposure', np.random.uniform(0.02, 0.12)),
        ('Training load monotony', np.random.uniform(-0.05, 0.10)),
    ]

    # Sort by absolute impact
    top_factors = sorted(all_factors, key=lambda x: abs(x[1]), reverse=True)[:5]

    # Confidence based on model agreement (mock)
    confidence_score = np.random.uniform(0.7, 0.95)
    if confidence_score >= 0.9:
        confidence = "very-high"
    elif confidence_score >= 0.8:
        confidence = "high"
    elif confidence_score >= 0.7:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        'risk_probability': risk_prob,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'severity_days': severity_days,
        'severity_level': severity_level,
        'archetype': archetype,
        'top_factors': top_factors,
        'confidence': confidence,
        'confidence_score': confidence_score,
        'ensemble_probs': {
            'catboost': risk_prob + np.random.normal(0, 0.02),
            'lightgbm': risk_prob + np.random.normal(0, 0.03),
            'xgboost': risk_prob + np.random.normal(0, 0.025),
        }
    }

def get_archetype_description(archetype):
    """Return archetype profile information."""
    profiles = {
        "High-Risk Frequent": {
            "description": "Accumulates micro-trauma quickly; sensitive to repeated intense loading.",
            "training_focus": "Reduce high-intensity exposure; recovery emphasis.",
        },
        "Moderate-Load High-Variance": {
            "description": "Load fluctuates widely; inconsistency reduces adaptive response.",
            "training_focus": "Stabilize weekly load variability.",
        },
        "Moderate-Risk Recurrent": {
            "description": "Recurring injury patterns; requires controlled overload.",
            "training_focus": "Reinforcement work + continuity blocks.",
        },
        "Low-Severity Stable": {
            "description": "Stable physiological responder; handles load well.",
            "training_focus": "Maintain current progression structure.",
        },
        "Catastrophic + Re-aggravation": {
            "description": "Very vulnerable to relapse; low tissue tolerance.",
            "training_focus": "Avoid overload; high-control microcycles.",
        },
    }
    return profiles.get(archetype, {"description": "", "training_focus": ""})

def generate_recommendations(prediction, player_data):
    """Generate actionable recommendations."""
    risk = prediction['risk_level']
    severity = prediction['severity_level']
    archetype = prediction['archetype']

    recommendations = []

    # Training recommendations
    if risk == "High":
        recommendations.append("⚠️ **Reduce training load immediately** - Consider 20-30% reduction in high-intensity work")
        recommendations.append("🔄 **Increase recovery time** - Add extra recovery day between intense sessions")
    elif risk == "Moderate":
        recommendations.append("⚡ **Monitor workload carefully** - Avoid sudden increases in training intensity")
        recommendations.append("📊 **Track daily wellness markers** - RPE, soreness, fatigue levels")
    else:
        recommendations.append("✅ **Continue current training structure** - Player responding well to load")

    # Match minutes guidance
    if risk == "High":
        if archetype == "Catastrophic + Re-aggravation":
            recommendations.append("⏱️ **Match minutes**: Maximum 45-60 mins, avoid consecutive full matches")
        else:
            recommendations.append("⏱️ **Match minutes**: Limit to 60-70 mins where possible, rotate when feasible")
    elif risk == "Moderate":
        recommendations.append("⏱️ **Match minutes**: Manage to ~70-80 mins, avoid extra time if possible")
    else:
        recommendations.append("⏱️ **Match minutes**: Full availability under normal rotation")

    # Archetype-specific
    profile = get_archetype_description(archetype)
    if profile['training_focus']:
        recommendations.append(f"🎯 **Archetype focus**: {profile['training_focus']}")

    # Use case specific
    if risk == "High":
        recommendations.append("🏆 **Fantasy football**: Consider bench or transfer - high injury risk")
    elif risk == "Moderate":
        recommendations.append("🏆 **Fantasy football**: Monitor closely - potential rotation risk")
    else:
        recommendations.append("🏆 **Fantasy football**: Good pick - low injury concern")

    return recommendations

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("⚽ Football Injury Risk Predictor")
    st.markdown("### AI-Powered Injury Risk Assessment Tool")

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <h4>⚠️ IMPORTANT DISCLAIMER</h4>
        <p><strong>This is a Proof-of-Concept (PoC) demonstration only.</strong></p>
        <ul>
            <li><strong>Not medical advice:</strong> This tool is for educational and entertainment purposes only. It does not replace professional medical assessment.</li>
            <li><strong>No liability:</strong> Predictions are based on statistical models and should not be used as the sole basis for medical, team selection, or betting decisions.</li>
            <li><strong>Use cases:</strong> Suitable for fantasy football insights, amateur team planning, and educational exploration of sports analytics.</li>
            <li><strong>Professional use:</strong> If you represent a professional organization, please contact for validation and proper deployment.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    teams, positions = load_player_data()
    models = load_models()

    # Model status
    if models is None:
        st.info("ℹ️ **Demo Mode**: Using simulated predictions. Train models to enable real predictions.")

    # Sidebar - Player Input Form
    st.sidebar.header("Player Information")
    st.sidebar.markdown("Enter player details to assess injury risk")

    with st.sidebar.form("player_form"):
        player_name = st.text_input(
            "Player Name",
            value="Harry Kane",
            help="Enter the player's name"
        )

        age = st.slider(
            "Age",
            min_value=16,
            max_value=40,
            value=28,
            help="Player's current age"
        )

        position = st.selectbox(
            "Position",
            options=positions,
            index=9,  # Center Forward
            help="Primary playing position"
        )

        team = st.selectbox(
            "Team",
            options=['Other'] + teams,
            index=0,
            help="Current team"
        )

        if team == 'Other':
            team = st.text_input("Team Name", value="Bayern Munich")

        recent_matches = st.number_input(
            "Recent Matches Played (last 4 weeks)",
            min_value=0,
            max_value=20,
            value=8,
            help="Number of matches played in the last 4 weeks (proxy for acute workload)"
        )

        fifa_rating = st.slider(
            "FIFA Rating (Optional)",
            min_value=50,
            max_value=99,
            value=88,
            help="Player's FIFA rating - optional quality metric"
        )

        submit_button = st.form_submit_button("🔮 Predict Injury Risk", use_container_width=True)

    # Main content area
    if submit_button:
        # Collect player data
        player_data = {
            'name': player_name,
            'age': age,
            'position': position,
            'team': team,
            'recent_matches': recent_matches,
            'fifa_rating': fifa_rating
        }

        # Generate prediction
        with st.spinner("Analyzing player data..."):
            prediction = generate_mock_prediction(player_data)

        # =====================================================================
        # RESULTS DISPLAY
        # =====================================================================

        st.markdown("---")
        st.header(f"📊 Risk Assessment: {player_name}")

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Risk Level",
                value=prediction['risk_level'],
                help="Overall injury risk classification"
            )

        with col2:
            st.metric(
                label="Risk Probability",
                value=f"{prediction['risk_probability']:.1%}",
                help="Statistical probability of injury in next 4 weeks"
            )

        with col3:
            st.metric(
                label="Projected Severity",
                value=f"{prediction['severity_days']} days",
                help="Expected days lost if injury occurs"
            )

        with col4:
            st.metric(
                label="Model Confidence",
                value=prediction['confidence'].title(),
                help="Ensemble model agreement level"
            )

        st.markdown("---")

        # Detailed Risk Panel
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("🎯 Risk Analysis")

            # Risk level card
            risk_html = f"""
            <div class="risk-{prediction['risk_color']}">
                <h3>{prediction['risk_level']} Risk ({prediction['risk_probability']:.1%})</h3>
                <p><strong>Severity Classification:</strong> {prediction['severity_level']}</p>
                <p><strong>Expected Days Lost (if injured):</strong> {prediction['severity_days']} days</p>
            </div>
            """
            st.markdown(risk_html, unsafe_allow_html=True)

            # Ensemble Model Agreement
            st.markdown("#### 🤖 Ensemble Model Predictions")
            ensemble = prediction['ensemble_probs']

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("CatBoost", f"{ensemble['catboost']:.1%}")
            with col_b:
                st.metric("LightGBM", f"{ensemble['lightgbm']:.1%}")
            with col_c:
                st.metric("XGBoost", f"{ensemble['xgboost']:.1%}")

            st.caption(f"Model agreement score: {prediction['confidence_score']:.0%}")

            # Top Risk Factors
            st.markdown("#### 📈 Top Risk Factors")
            for i, (factor, impact) in enumerate(prediction['top_factors'], 1):
                direction = "🔴" if impact > 0 else "🟢"
                st.markdown(f"{i}. {direction} **{factor}**: {impact:+.3f}")

        with col_right:
            st.subheader("👤 Player Archetype")

            profile = get_archetype_description(prediction['archetype'])

            st.markdown(f"### {prediction['archetype']}")
            st.markdown(f"**Description:** {profile['description']}")
            st.markdown(f"**Training Focus:** {profile['training_focus']}")

            st.markdown("---")

            # Recommendations
            st.markdown("#### 💡 Recommendations")
            recommendations = generate_recommendations(prediction, player_data)

            for rec in recommendations:
                st.markdown(rec)

        # =====================================================================
        # ADDITIONAL SECTIONS
        # =====================================================================

        st.markdown("---")
        st.markdown("---")

        # Expandable sections
        with st.expander("ℹ️ How It Works", expanded=False):
            st.markdown("""
            ### Methodology

            This injury risk predictor uses a **machine learning ensemble approach** combining:

            1. **Classification Models** (CatBoost, LightGBM, XGBoost)
               - Predict probability of injury in next 4 weeks
               - Trained on historical injury and performance data

            2. **Regression Models** (Severity Prediction)
               - Estimate expected days lost if injury occurs
               - Based on injury type patterns and player characteristics

            3. **Clustering Analysis** (Player Archetypes)
               - HDBSCAN + K-Means hybrid clustering
               - Identifies 5 distinct injury risk profiles

            4. **SHAP Analysis** (Explainability)
               - Identifies key risk factors for each prediction
               - Provides interpretable insights for decision-making

            ### Key Features

            - **Workload metrics**: Acute and chronic load ratios
            - **Age & position**: Injury risk varies by role and age
            - **Performance data**: FIFA ratings, match statistics
            - **Historical patterns**: Previous injury frequency and severity
            - **Recovery metrics**: Time between injuries, load variability
            """)

        with st.expander("📊 Data Sources", expanded=False):
            st.markdown("""
            ### Training Data

            - **Player Statistics**: FIFA dataset (1992-2025) - 118+ features per player
            - **Injury Records**: Premier League injury database with ~2000+ injury events
            - **Match Data**: Performance metrics, goals, assists, minutes played
            - **Temporal Features**: Rolling averages, workload trends, seasonal patterns

            ### Data Coverage

            - **Leagues**: Primarily Premier League, extendable to other leagues
            - **Seasons**: 1992-2025 historical data
            - **Players**: Thousands of professional footballers
            - **Injuries**: Multiple injury types (hamstring, knee, ankle, etc.)

            ### Privacy & Ethics

            - All data is publicly available or aggregated
            - No personal medical information is used
            - Predictions are statistical, not deterministic
            """)

        with st.expander("⚠️ Limitations & Caveats", expanded=False):
            st.markdown("""
            ### Model Limitations

            1. **Data Quality**
               - Relies on publicly available injury data which may be incomplete
               - Injury dates and return dates are approximations
               - Not all minor injuries are reported publicly

            2. **Scope**
               - Trained primarily on Premier League data
               - May not generalize perfectly to other leagues or levels
               - Individual medical history not included (privacy)

            3. **Temporal Limitations**
               - Cannot account for unreported factors (sleep, nutrition, stress)
               - Does not include real-time GPS/accelerometer data
               - Training load is proxied from match minutes, not actual load

            4. **Statistical Nature**
               - Predictions are probabilities, not certainties
               - High-risk players may stay healthy; low-risk may get injured
               - Should be one input among many for decision-making

            ### Not Suitable For

            - ❌ Medical diagnosis or treatment decisions
            - ❌ Sole basis for team selection or transfer decisions
            - ❌ Betting or gambling purposes
            - ❌ Professional deployment without validation

            ### Recommended Use

            - ✅ Fantasy football team planning
            - ✅ Amateur team rotation guidance
            - ✅ Educational exploration of sports analytics
            - ✅ Conversation starter for injury prevention strategies
            """)

        with st.expander("🎯 Use Cases & Applications", expanded=False):
            st.markdown("""
            ### Fantasy Football

            - **Transfer Planning**: Avoid high-risk players before busy periods
            - **Captain Selection**: Choose low-risk captains for double gameweeks
            - **Bench Strategy**: Keep high-risk players on bench during tough fixtures

            ### Amateur & Semi-Pro Teams

            - **Training Load Management**: Adjust intensity based on risk profiles
            - **Squad Rotation**: Rest high-risk players proactively
            - **Injury Prevention Programs**: Target interventions for vulnerable players

            ### Sports Analytics & Research

            - **Injury Pattern Analysis**: Understand position-specific risks
            - **Workload Optimization**: Test different training load strategies
            - **Performance Impact**: Correlate injury risk with performance decline

            ### Coaching & Sports Science

            - **Conversation Tool**: Start discussions about load management
            - **Data-Driven Decisions**: Complement expert judgment with statistical insights
            - **Player Education**: Help players understand their injury risk factors

            ### Professional Deployment

            For professional clubs and organizations:

            1. **Validation Required**: Test against your specific population and data
            2. **Custom Training**: Retrain models on your proprietary data (GPS, medical records)
            3. **Integration**: Embed into existing sports science platforms
            4. **Expert Review**: Always combine model output with medical professional judgment

            *Contact for professional deployment consultation*
            """)

    else:
        # Initial state - show overview
        st.markdown("---")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            ### Welcome to the Injury Risk Predictor

            This tool uses machine learning to predict football player injury risk based on:

            - **Player characteristics** (age, position, team)
            - **Workload metrics** (recent matches played as proxy)
            - **Historical patterns** (learned from thousands of injury events)

            #### How to Use

            1. Enter player details in the sidebar form
            2. Click "Predict Injury Risk" button
            3. Review the comprehensive risk assessment
            4. Explore recommendations and insights

            #### What You'll Get

            - **Risk probability**: Statistical likelihood of injury
            - **Severity prediction**: Expected days lost if injured
            - **Player archetype**: Risk profile classification
            - **Top risk factors**: Key drivers of the prediction
            - **Actionable recommendations**: Training, match minutes, and recovery guidance

            ⬅️ **Get started by filling out the form on the left!**
            """)

        with col2:
            st.info("""
            **Quick Start Examples**

            Try these profiles:

            1. **Young winger** (age 22, 10+ recent matches)
               → High workload risk

            2. **Veteran striker** (age 35+)
               → Age-related risk

            3. **Goalkeeper** (any age, low matches)
               → Low risk profile
            """)

        # Show feature highlights
        st.markdown("---")
        st.markdown("### Key Features")

        feat1, feat2, feat3 = st.columns(3)

        with feat1:
            st.markdown("""
            #### 🤖 Ensemble ML
            Three gradient boosting models work together for robust predictions
            """)

        with feat2:
            st.markdown("""
            #### 👥 Player Archetypes
            5 distinct risk profiles based on clustering analysis
            """)

        with feat3:
            st.markdown("""
            #### 📊 Explainable AI
            SHAP analysis shows which factors drive each prediction
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Injury Risk Predictor v1.0 - Proof of Concept</p>
        <p style='font-size: 12px;'>
            Built with Streamlit • Machine Learning Models: CatBoost, LightGBM, XGBoost •
            Clustering: HDBSCAN + K-Means • Explainability: SHAP
        </p>
        <p style='font-size: 12px;'>
            ⚠️ For educational and entertainment purposes only • Not medical advice
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
