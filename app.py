from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATHS = [
    ROOT / "models" / "saved" / "national_team_model.pkl",
    ROOT / "models" / "national_team_model.pkl",
]


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="International Football Predictor",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 International Football Predictor")
st.caption(
    "National-team Elo + Dixon-Coles model · "
    "1X2 · Goals · BTTS · Exact Scores"
)


# =============================================================================
# LOAD MODEL CODE
# =============================================================================

# Make repository root importable.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    from models.national_team_model import predict
except Exception as exc:
    st.error("Could not import the national-team model.")
    st.code(str(exc))
    st.stop()


# =============================================================================
# LOAD TRAINED MODEL
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_model():
    model_path = next(
        (path for path in MODEL_PATHS if path.exists()),
        None,
    )

    if model_path is None:
        raise FileNotFoundError(
            "Trained model not found.\n\n"
            "Expected one of:\n"
            "models/saved/national_team_model.pkl\n"
            "models/national_team_model.pkl"
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model, model_path


try:
    model, model_path = load_model()
except Exception as exc:
    st.error("The trained national-team model could not be loaded.")
    st.warning(
        "The Streamlit interface is ready, but the trained .pkl model "
        "must also be present in the GitHub repository."
    )
    st.code(str(exc))
    st.stop()


# =============================================================================
# TEAM LIST
# =============================================================================

def get_team_names(model):
    """
    Extract team names from the trained model.
    """

    if isinstance(model, dict):

        # Preferred source from the current model.
        name_to_id = model.get("name_to_id")

        if isinstance(name_to_id, dict) and name_to_id:
            return sorted(
                str(name).strip()
                for name in name_to_id.keys()
                if str(name).strip()
            )

        # Fallback.
        names = model.get("names")

        if isinstance(names, (list, tuple, set)):
            return sorted(
                str(name).strip()
                for name in names
                if str(name).strip()
            )

    raise ValueError(
        "Could not find national-team names inside the trained model."
    )


try:
    teams = get_team_names(model)
except Exception as exc:
    st.error("Could not read the national-team list from the model.")
    st.code(str(exc))
    st.stop()


if len(teams) < 2:
    st.error("The model contains fewer than two national teams.")
    st.stop()


# =============================================================================
# TEAM ID RESOLUTION
# =============================================================================

def get_team_id(model, team_name):
    """
    Convert displayed team name into the ID expected by predict().
    """

    if isinstance(model, dict):

        name_to_id = model.get("name_to_id")

        if isinstance(name_to_id, dict):

            if team_name in name_to_id:
                return name_to_id[team_name]

            # Case-insensitive fallback.
            lowered = team_name.lower()

            for name, team_id in name_to_id.items():
                if str(name).lower() == lowered:
                    return team_id

    raise ValueError(
        f"Could not resolve team ID for: {team_name}"
    )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.header("⚽ Match Setup")

    home_team = st.selectbox(
        "Home team",
        teams,
        index=0,
    )

    away_options = [
        team
        for team in teams
        if team != home_team
    ]

    away_team = st.selectbox(
        "Away team",
        away_options,
        index=0,
    )

    neutral = st.checkbox(
        "Neutral venue",
        value=True,
        help="Enable this for matches played at a neutral venue.",
    )

    st.divider()

    predict_button = st.button(
        "🔮 RUN PREDICTION",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.caption(f"Teams available: {len(teams)}")
    st.caption(f"Model: {model_path.name}")


# =============================================================================
# INITIAL STATE
# =============================================================================

if not predict_button:

    st.info(
        "Select two national teams and press **RUN PREDICTION**."
    )

    st.markdown(
        """
### Markets available

**Match result**
- Home win
- Draw
- Away win

**Goals**
- Over / Under 0.5
- Over / Under 1.5
- Over / Under 2.5
- Over / Under 3.5
- Over / Under 4.5

**Other**
- BTTS Yes / No
- Expected goals
- Most likely exact scores
- Elo ratings
"""
    )

    st.stop()


# =============================================================================
# VALIDATION
# =============================================================================

if home_team == away_team:

    st.error("Home and away teams must be different.")
    st.stop()


# =============================================================================
# RUN MODEL
# =============================================================================

with st.spinner(
    f"Calculating {home_team} vs {away_team}..."
):

    try:

        home_id = get_team_id(
            model,
            home_team,
        )

        away_id = get_team_id(
            model,
            away_team,
        )

        prediction = predict(
            model,
            home_id,
            away_id,
            neutral=neutral,
        )

    except Exception as exc:

        st.error("Prediction failed.")
        st.exception(exc)
        st.stop()


# =============================================================================
# EXTRACT BASIC OUTPUT
# =============================================================================

home_probability = float(
    prediction.get("home_win", 0.0)
)

draw_probability = float(
    prediction.get("draw", 0.0)
)

away_probability = float(
    prediction.get("away_win", 0.0)
)

xg_home = prediction.get(
    "lambda_home",
    prediction.get("xg_home", None),
)

xg_away = prediction.get(
    "lambda_away",
    prediction.get("xg_away", None),
)

goals_markets = prediction.get(
    "goals_over_under",
    {},
)

btts_yes = prediction.get(
    "btts_yes",
    None,
)

btts_no = prediction.get(
    "btts_no",
    None,
)

top_scores = prediction.get(
    "top_scores",
    [],
)


# =============================================================================
# HEADER
# =============================================================================

st.header(
    f"{home_team} vs {away_team}"
)

if neutral:
    st.caption("🏟️ Neutral venue")
else:
    st.caption("🏠 Home venue")


# =============================================================================
# 1X2
# =============================================================================

st.subheader("1X2")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Home",
        f"{home_probability:.1%}",
    )

with c2:
    st.metric(
        "Draw",
        f"{draw_probability:.1%}",
    )

with c3:
    st.metric(
        "Away",
        f"{away_probability:.1%}",
    )


# =============================================================================
# EXPECTED GOALS
# =============================================================================

st.subheader("Expected Goals")

x1, x2 = st.columns(2)

with x1:

    if xg_home is not None:
        st.metric(
            home_team,
            f"{float(xg_home):.2f} xG",
        )
    else:
        st.metric(
            home_team,
            "N/A",
        )

with x2:

    if xg_away is not None:
        st.metric(
            away_team,
            f"{float(xg_away):.2f} xG",
        )
    else:
        st.metric(
            away_team,
            "N/A",
        )


# =============================================================================
# GOALS MARKETS
# =============================================================================

st.subheader("⚽ Total Goals")

if goals_markets:

    goal_rows = []

    for line in [
        "0.5",
        "1.5",
        "2.5",
        "3.5",
        "4.5",
    ]:

        market = goals_markets.get(line)

        if not market:
            continue

        over = float(
            market.get("over", 0.0)
        )

        under = float(
            market.get("under", 0.0)
        )

        goal_rows.append(
            {
                "Line": line,
                "Over": f"{over:.1%}",
                "Under": f"{under:.1%}",
            }
        )

    if goal_rows:

        st.dataframe(
            pd.DataFrame(goal_rows),
            use_container_width=True,
            hide_index=True,
        )

else:

    st.warning(
        "The loaded model does not contain the new goals_over_under output."
    )

    # Compatibility fallback for older model code.
    if "over_2_5" in prediction:

        st.write(
            f"Over 2.5: "
            f"{float(prediction['over_2_5']):.1%}"
        )

    if "under_2_5" in prediction:

        st.write(
            f"Under 2.5: "
            f"{float(prediction['under_2_5']):.1%}"
        )


# =============================================================================
# BTTS
# =============================================================================

st.subheader("Both Teams To Score")

b1, b2 = st.columns(2)

with b1:

    if btts_yes is not None:
        st.metric(
            "BTTS — Yes",
            f"{float(btts_yes):.1%}",
        )
    else:
        st.metric(
            "BTTS — Yes",
            "N/A",
        )

with b2:

    if btts_no is not None:
        st.metric(
            "BTTS — No",
            f"{float(btts_no):.1%}",
        )
    else:
        st.metric(
            "BTTS — No",
            "N/A",
        )


# =============================================================================
# EXACT SCORES
# =============================================================================

st.subheader("🎯 Most Likely Exact Scores")

if top_scores:

    score_rows = []

    for score, probability in top_scores:

        score_rows.append(
            {
                "Score": str(score).replace(
                    ":",
                    "–",
                ),
                "Probability": f"{float(probability):.1%}",
            }
        )

    st.dataframe(
        pd.DataFrame(score_rows),
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Exact-score probabilities were not returned by the model."
    )


# =============================================================================
# ELO
# =============================================================================

elo_home = prediction.get(
    "elo_home",
    None,
)

elo_away = prediction.get(
    "elo_away",
    None,
)

if elo_home is not None or elo_away is not None:

    st.subheader("📈 Elo")

    e1, e2 = st.columns(2)

    with e1:

        if elo_home is not None:
            st.metric(
                home_team,
                f"{float(elo_home):.0f}",
            )

    with e2:

        if elo_away is not None:
            st.metric(
                away_team,
                f"{float(elo_away):.0f}",
            )


# =============================================================================
# RAW MODEL OUTPUT
# =============================================================================

with st.expander("Model details"):

    st.json(
        {
            key: value
            for key, value in prediction.items()
            if key != "score_matrix"
        }
    )

st.divider()

st.caption(
    "Probabilities are model estimates generated from the "
    "national-team Elo + Dixon-Coles model. They are not guarantees."
)