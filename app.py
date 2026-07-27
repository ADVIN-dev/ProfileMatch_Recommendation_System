from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from users import create_users_csv

st.set_page_config(
    page_title="ProfileMatch: Intelligent User Recommendation System",
    page_icon="👥",
    layout="wide",
)

PROJECT_FOLDER = Path(__file__).resolve().parent
CSV_FILE = PROJECT_FOLDER / "users.csv"


# -------------------------------------------------------------------
# MBTI compatibility rules used for this student project
# -------------------------------------------------------------------
MBTI_COMPATIBILITY = {
    "INTJ": ["ENFP", "ENTP", "INFJ"],
    "INTP": ["ENTJ", "ENFJ", "ENTP"],
    "ENTJ": ["INTP", "INFP", "ENTP"],
    "ENTP": ["INFJ", "INTJ", "ENFP"],
    "INFJ": ["ENTP", "ENFP", "INTJ"],
    "INFP": ["ENFJ", "ENTJ", "INFJ"],
    "ENFJ": ["INFP", "ISFP", "INTP"],
    "ENFP": ["INTJ", "INFJ", "ENTP"],
    "ISTJ": ["ESFP", "ESTP", "ISFJ"],
    "ISFJ": ["ESFP", "ESTP", "ISTJ"],
    "ESTJ": ["ISFP", "ISTP", "ESFJ"],
    "ESFJ": ["ISFP", "ISTP", "ESTJ"],
    "ISTP": ["ESFJ", "ESTJ", "ISFP"],
    "ISFP": ["ENFJ", "ESFJ", "ESTJ"],
    "ESTP": ["ISFJ", "ISTJ", "ESFP"],
    "ESFP": ["ISFJ", "ISTJ", "ESTP"],
}


def validate_dataset(df: pd.DataFrame) -> None:
    """Check that the dataset contains all required columns."""

    required_columns = {
        "user_id",
        "name",
        "location",
        "skillset",
        "income",
        "mbti",
        "about_me",
        "professional_summary",
        "career_goal",
        "experience",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"The dataset is missing these columns: {missing_text}")

    valid_mbti_types = set(MBTI_COMPATIBILITY)

    invalid_types = sorted(set(df["mbti"].astype(str).str.upper()) - valid_mbti_types)

    if invalid_types:
        invalid_text = ", ".join(invalid_types)
        raise ValueError(f"Invalid MBTI values found: {invalid_text}")


@st.cache_data
def load_data() -> pd.DataFrame:
    """Create users.csv when needed and load it."""

    create_users_csv(force=False)

    dataframe = pd.read_csv(CSV_FILE)

    dataframe["name"] = dataframe["name"].astype(str).str.strip()
    dataframe["location"] = dataframe["location"].astype(str).str.strip()
    dataframe["skillset"] = dataframe["skillset"].fillna("").astype(str)
    dataframe["mbti"] = dataframe["mbti"].astype(str).str.upper().str.strip()
    dataframe["income"] = pd.to_numeric(dataframe["income"], errors="coerce")

    dataframe["income"] = dataframe["income"].fillna(dataframe["income"].median())
    dataframe["profile_text"] = (
        dataframe["about_me"].fillna("")
        + " "
        + dataframe["professional_summary"].fillna("")
        + " "
        + dataframe["career_goal"].fillna("")
        + " "
        + dataframe["skillset"].fillna("")
    )

    validate_dataset(dataframe)

    return dataframe


def calculate_mbti_compatibility(
    first_mbti: str, second_mbti: str
) -> tuple[float, str]:
    """
    Return an MBTI compatibility score and explanation.

    This is a project-defined recommendation rule, not a clinical test.
    """

    first_mbti = first_mbti.upper().strip()
    second_mbti = second_mbti.upper().strip()

    if first_mbti == second_mbti:
        return 100.0, "Same MBTI type"

    first_matches_second = second_mbti in MBTI_COMPATIBILITY.get(first_mbti, [])

    second_matches_first = first_mbti in MBTI_COMPATIBILITY.get(second_mbti, [])

    if first_matches_second or second_matches_first:
        return 90.0, "Highly compatible MBTI types"

    matching_letters = sum(
        first_letter == second_letter
        for first_letter, second_letter in zip(first_mbti, second_mbti)
    )

    scores = {
        3: (75.0, "Three MBTI preferences match"),
        2: (50.0, "Two MBTI preferences match"),
        1: (25.0, "One MBTI preference matches"),
        0: (0.0, "No MBTI preferences match"),
    }

    return scores.get(matching_letters, (0.0, "Low MBTI similarity"))


def train_knn_model(df: pd.DataFrame) -> tuple[NearestNeighbors, object]:
    """
    Train an unsupervised K-Nearest Neighbors model.

    The ML model uses:
    1. TF-IDF Profile features
    2. Standardized income

    The profile text includes:
    • About Me
    • Professional Summary
    • Career Goal
    • Skillset

    MBTI compatibility is calculated separately and combined with
    profile similarity to generate the final recommendation score.
    """

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))

    profile_features = vectorizer.fit_transform(df["profile_text"])

    scaler = StandardScaler()

    income_features = scaler.fit_transform(df[["income"]])

    combined_features = hstack([profile_features, csr_matrix(income_features)])

    model = NearestNeighbors(metric="cosine", algorithm="brute")

    model.fit(combined_features)

    return model, combined_features


def get_recommendations(
    df: pd.DataFrame, selected_name: str, number_of_recommendations: int
) -> pd.DataFrame:
    """
    Recommend users using MBTI compatibility and KNN similarity.

    Final score:
    - 60% NLP Profile Similarity
    - 25% MBTI Compatibility
    - 10% Location Match
    - 5% Income Similarity
    """

    model, feature_matrix = train_knn_model(df)

    selected_indices = df.index[df["name"] == selected_name].tolist()

    if not selected_indices:
        raise ValueError(f"User '{selected_name}' was not found.")

    selected_index = selected_indices[0]
    selected_user = df.loc[selected_index]

    neighbours_to_fetch = min(len(df), number_of_recommendations + 1)

    distances, indices = model.kneighbors(
        feature_matrix[selected_index], n_neighbors=neighbours_to_fetch
    )

    recommendations = []

    for distance, neighbour_index in zip(distances[0], indices[0]):
        if neighbour_index == selected_index:
            continue

        candidate = df.iloc[neighbour_index]

        knn_similarity = float(np.clip((1 - distance) * 100, 0, 100))

        mbti_score, mbti_reason = calculate_mbti_compatibility(
            selected_user["mbti"], candidate["mbti"]
        )

        location_score = (
            100.0
            if selected_user["location"].casefold() == candidate["location"].casefold()
            else 0.0
        )

        income_score = max(0,100-abs(selected_user["income"]-candidate["income"])*5)

        final_score=(
            0.60*knn_similarity+
            0.25*mbti_score+
            0.10*location_score+
            0.05*income_score
        )

        reason = []

        if mbti_score >= 90:
            reason.append("Highly Compatible MBTI")

        if location_score == 100:
            reason.append("Same Location")

        if knn_similarity >= 80:
            reason.append("Very Similar Profile")

        if not reason:
            reason.append("Similar User Profile")

        recommendations.append(
            {
                "Name": candidate["name"],
                "MBTI": candidate["mbti"],
                "Experience": candidate["experience"],
                "Location": candidate["location"],
                "Skills": candidate["skillset"],
                "Reason": ", ".join(reason),
                "Income (LPA)": round(float(candidate["income"]), 2),
                "MBTI Score": round(mbti_score, 2),
                "KNN Similarity": round(knn_similarity, 2),
                "Location Score": round(location_score, 2),
                "Final Score": round(final_score, 2),
            }
        )

    result = pd.DataFrame(recommendations)

    if result.empty:
        return result

    return (
        result.sort_values(
            by=["Final Score", "MBTI Score", "KNN Similarity"], ascending=False
        )
        .head(number_of_recommendations)
        .reset_index(drop=True)
    )


def main() -> None:
    st.title("👥 ProfileMatch")
    st.subheader("Intelligent User Recommendation System")

    st.write("""
        This project recommends similar users using:

        • NLP-based Profile Similarity (TF-IDF)

        • K-Nearest Neighbors (KNN)

        • MBTI Compatibility

        • Location Matching

        • Income Similarity

        The system generates personalized recommendations using a hybrid scoring approach.
        """)

    st.info("""
        Recommendation Score

        60% Profile Similarity (TF-IDF + KNN)

        25% MBTI Compatibility

        10% Location Match

        5% Income Similarity
        """)

    try:
        df = load_data()
    except Exception as error:
        st.error(f"Unable to load the dataset: {error}")
        st.stop()

    st.sidebar.header("Recommendation Settings")

    selected_name = st.sidebar.selectbox("Select a user", options=df["name"].tolist())

    maximum_recommendations = max(1, len(df) - 1)

    number_of_recommendations = st.sidebar.slider(
        "Number of recommendations",
        min_value=1,
        max_value=maximum_recommendations,
        value=min(5, maximum_recommendations),
    )

    show_dataset = st.sidebar.checkbox("Show complete dataset")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Statistics")
    st.sidebar.write(f"Total Users: {len(df)}")
    st.sidebar.write(f"Total Cities: {df['location'].nunique()}")
    st.sidebar.write(f"MBTI Types: {df['mbti'].nunique()}")
    st.sidebar.write(f"Average Income: {df['income'].mean():.2f} LPA")

    selected_user = df.loc[df["name"] == selected_name].iloc[0]

    st.subheader("Selected User Profile")

    col1, col2, col3 = st.columns(3)

    col1.metric("Name", selected_user["name"])
    col2.metric("MBTI", selected_user["mbti"])
    col3.metric("Income", f'{selected_user["income"]:.1f} LPA')

    st.write("### Profile Information")

    st.write("**Experience:**", selected_user["experience"])

    st.write("**Location:**", selected_user["location"])

    st.write("**Skills:**", selected_user["skillset"])

    st.write("**About Me:**")

    st.info(selected_user["about_me"])

    st.write("**Professional Summary:**")

    st.success(selected_user["professional_summary"])

    st.write("**Career Goal:**")

    st.warning(selected_user["career_goal"])

    if st.button("Find Best Matches", type="primary", use_container_width=True):
        try:
            recommendations = get_recommendations(
                df=df,
                selected_name=selected_name,
                number_of_recommendations=number_of_recommendations,
            )
        except Exception as error:
            st.error(f"Recommendation failed: {error}")
            st.stop()

        if recommendations.empty:
            st.warning("No recommendations were found.")
        else:
            st.subheader(f"Top {len(recommendations)} Matches")

            st.dataframe(recommendations, use_container_width=True, hide_index=True)

            st.subheader("Final Score Comparison")

            chart_data = recommendations.set_index("Name")[["Final Score"]]

            st.bar_chart(chart_data)

            st.subheader("MBTI Distribution")

            st.bar_chart(df["mbti"].value_counts())

            st.subheader("Most Common Skills")

            skills=df["skillset"].str.replace("[","",regex=False)\
            .str.replace("]","",regex=False)\
            .str.replace("'","",regex=False)\
            .str.split(",")

            from collections import Counter

            counter=Counter()

            for row in skills:
                counter.update([i.strip() for i in row])

            skill_df=pd.DataFrame(
            counter.items(),
            columns=["Skill","Count"]
            ).sort_values("Count",ascending=False)

            st.bar_chart(skill_df.set_index("Skill"))

            best_match = recommendations.iloc[0]

            st.success(
                f"Best match: {best_match['Name']} "
                f"({best_match['MBTI']}) — "
                f"{best_match['Final Score']}%"
            )

    if show_dataset:
        st.subheader("Complete User Dataset")

        st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("How does the KNN model work?"):
        st.write("""
            The KNN model converts each user's complete profile into numerical vectors using TF-IDF.

            The profile includes:

            • About Me

            • Professional Summary

            • Career Goal

            • Skillset

            KNN then finds users with the most similar profiles.
            MBTI compatibility, location, and income similarity are combined with profile similarity to generate the final recommendation score.
            """)

    with st.expander("Project Workflow"):

        st.markdown("""
            User Profile

            ↓

            Text Preprocessing

            ↓

            TF-IDF Vectorization

            ↓

            KNN Similarity
            
            ↓

            MBTI Compatibility

            ↓

            Hybrid Score Calculation

            ↓

            Top Recommended Users
            """)
    with st.expander("Important note about MBTI"):
        st.write("""
            The MBTI compatibility table in this application is a
            project-defined heuristic. It is suitable for demonstrating
            a recommendation system, but it should not be treated as a
            scientifically validated personality assessment.
            """)

st.markdown("---")

st.caption("""
Developed by Aditya Raj

B.Tech Major Project

Department of Computer Science

KIIT University, Bhubaneswar
""")

if __name__ == "__main__":
    main()
