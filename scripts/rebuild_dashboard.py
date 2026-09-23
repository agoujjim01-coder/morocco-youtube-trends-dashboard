
import os
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from langdetect import detect

from utils.config import DATA_DIR


# -------------------------------------------------
# Text helpers
# -------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def detect_language(text):

    try:
        return detect(str(text))

    except Exception:
        return "unknown"


def simple_processed_text(text):

    """
    Lightweight multilingual cleanup for clustering.

    We intentionally keep this simple and reproducible
    for the automatic weekly pipeline.
    """

    words = str(text).split()

    generic_stopwords = {
        # English
        "the", "and", "you", "that", "this", "with",
        "for", "are", "was", "have", "has", "just",
        "like", "video", "videos", "channel",
        "subscribe", "guys", "really", "going",

        # French
        "les", "des", "une", "dans", "pour", "avec",
        "mais", "plus", "tout", "faire", "fait",
        "comme", "bien",

        # Arabic / Darija
        "هذا", "هذه", "انا", "أنا", "هو", "هي",
        "نحن", "على", "الى", "إلى", "من", "في",
        "اللي", "يعني", "الله", "هاد", "ديال",
        "راه", "باش", "حيت", "غير", "دابا",
        "واش", "فين", "كيفاش", "فيديو", "الفيديو"
    }

    words = [
        word
        for word in words
        if len(word) > 2
        and word not in generic_stopwords
    ]

    return " ".join(words)


# -------------------------------------------------
# Main dashboard rebuild
# -------------------------------------------------

def rebuild_dashboard():

    dashboard_path = os.path.join(
        DATA_DIR,
        "dashboard_data.csv"
    )

    weekly_path = os.path.join(
        DATA_DIR,
        "weekly_analyzed_youtube.csv"
    )

    # ---------------------------------------------
    # Load existing dashboard history
    # ---------------------------------------------

    if os.path.exists(dashboard_path):

        old_df = pd.read_csv(
            dashboard_path
        )

    else:

        old_df = pd.DataFrame()

    # ---------------------------------------------
    # Load this week's analysed batch
    # ---------------------------------------------

    if os.path.exists(weekly_path):

        weekly_df = pd.read_csv(
            weekly_path
        )

    else:

        print(
            "No weekly analysed file found. "
            "Dashboard remains unchanged."
        )

        return old_df

    print(
        "Existing dashboard videos:",
        len(old_df)
    )

    print(
        "Weekly analysed videos:",
        len(weekly_df)
    )

    # ---------------------------------------------
    # Combine datasets
    # ---------------------------------------------
    # Preserve existing values when weekly metadata is missing
    if not old_df.empty and not weekly_df.empty:

        old_df["video_id"] = old_df["video_id"].astype(str)
        weekly_df["video_id"] = weekly_df["video_id"].astype(str)

        old_by_id = (
            old_df
            .drop_duplicates(subset=["video_id"], keep="last")
            .set_index("video_id")
        )

        weekly_df = weekly_df.set_index("video_id")
        
        weekly_df = weekly_df.replace(
            r"^\s*$",
            pd.NA,
            regex=True
        )

        weekly_df = weekly_df.combine_first(
            old_by_id
        ).loc[weekly_df.index].reset_index()
        
    combined = pd.concat(
        [old_df, weekly_df],
        ignore_index=True,
        sort=False
    )

    if "video_id" not in combined.columns:
        raise ValueError(
            "video_id column is missing."
        )

    combined["video_id"] = (
        combined["video_id"]
        .astype(str)
    )

    # Prefer a row with a transcript when the same video
    # appears in both the existing and weekly datasets.
    # If both rows have transcripts (or neither does),
    # keep the newer weekly row.

    combined["_has_usable_transcript"] = (
        combined["transcript"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    combined = (
        combined
        .sort_values(
            "_has_usable_transcript",
            kind="stable"
        )
        .drop_duplicates(
            subset=["video_id"],
            keep="last"
        )
        .drop(columns=["_has_usable_transcript"])
        .reset_index(drop=True)
    )

    # ---------------------------------------------
    # Preserve videos with and without transcripts
    # ---------------------------------------------

    combined["transcript"] = (
        combined["transcript"]
        .fillna("")
        .astype(str)
    )

    combined["has_transcript"] = (
        combined["transcript"]
        .str.strip()
        .ne("")
    )

    combined.reset_index(
        drop=True,
        inplace=True
    )
    
    # ---------------------------------------------
    # Rebuild Phase 1 columns
    # ---------------------------------------------

    combined["clean_transcript"] = (
        combined["transcript"]
        .apply(clean_text)
    )

    combined["language"] = (
        combined["clean_transcript"]
        .apply(
            lambda text: detect_language(text)
            if str(text).strip()
            else "unknown"
        )
    )

    # Count words only when a transcript exists
    combined["transcript_word_count"] = (
        combined["transcript"]
        .str.split()
        .str.len()
        .where(combined["has_transcript"])
    )

    # Keep compatibility with the dashboard
    combined["transcript_length"] = (
        combined["transcript_word_count"]
    )

    combined["word_count"] = (
        combined["clean_transcript"]
        .str.split()
        .str.len()
    )

    combined["processed_text"] = (
        combined["clean_transcript"]
        .apply(simple_processed_text)
    )

    # ---------------------------------------------
    # Topic clustering
    # ---------------------------------------------

    analysis_text = (
        combined["title_caption"]
        .fillna("")
        .astype(str)
        + " "
        + combined["processed_text"]
        .fillna("")
        .astype(str)
    )

    if len(combined) >= 6:

        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8
        )

        X = vectorizer.fit_transform(
            analysis_text
        )

        k = 6

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        combined["topic"] = (
            kmeans.fit_predict(X)
        )

        # -----------------------------------------
        # Create dynamic cluster descriptions
        # -----------------------------------------

        terms = (
            vectorizer
            .get_feature_names_out()
        )

        order = (
            kmeans
            .cluster_centers_
            .argsort()[:, ::-1]
        )

        topic_labels = {}

        for topic_id in range(k):

            top_terms = [
                terms[ind]
                for ind in order[
                    topic_id, :3
                ]
            ]

            topic_labels[topic_id] = (
                " / ".join(top_terms)
            )

        combined["topic_label"] = (
            combined["topic"]
            .map(topic_labels)
        )

    else:

        combined["topic"] = 0
        combined["topic_label"] = "Insufficient data"

    # ---------------------------------------------
    # Add content categories from Gemini analysis
    # ---------------------------------------------

    ai_path = os.path.join(
        DATA_DIR,
        "gemini_analysis.csv"
    )

    if os.path.exists(ai_path):

        ai_df = pd.read_csv(ai_path)

        if "content_category" in ai_df.columns:

            category_lookup = (
                ai_df
                .dropna(subset=["video_id"])
                .drop_duplicates(
                    subset=["video_id"],
                    keep="last"
                )
                .set_index(
                    ai_df.dropna(subset=["video_id"])
                    .drop_duplicates(
                        subset=["video_id"],
                        keep="last"
                    )["video_id"].astype(str)
                )["content_category"]
            )

            combined["content_category"] = (
                combined["video_id"]
                .astype(str)
                .map(category_lookup)
                .fillna(
                    combined.get(
                        "content_category",
                        pd.Series(index=combined.index, dtype="object")
                    )
                )
                .fillna("Other / Unclear")
            )

    # ---------------------------------------------
    # Ensure expected columns exist
    # ---------------------------------------------

    expected_columns = [
        "platform",
        "content_category",
        "video_id",
        "url",
        "title_caption",
        "hashtags",
        "views",
        "likes",
        "comments",
        "channel_account",
        "download_path",
        "scraped_at",
        "transcript",
        "clean_transcript",
        "language",
        "transcript_word_count",
        "transcript_length",
        "word_count",
        "processed_text",
        "topic",
        "topic_label"
    ]

    for column in expected_columns:

        if column not in combined.columns:
            combined[column] = ""

    # Keep expected columns first, but preserve any extra
    # columns already present in the dashboard dataset.
    extra_columns = [
        col for col in combined.columns
        if col not in expected_columns
    ]

    combined = combined[
        expected_columns + extra_columns
    ]

    # ---------------------------------------------
    # Save dashboard dataset
    # ---------------------------------------------

    combined.to_csv(
        dashboard_path,
        index=False
    )

    print("\nDashboard rebuilt ✅")
    print(
        "Total usable videos:",
        len(combined)
    )

    print(
        "Unique video IDs:",
        combined["video_id"].nunique()
    )

    print("\nVideos per topic:")
    print(
        combined["topic"]
        .value_counts()
        .sort_index()
    )

    print("\nDynamic topic labels:")

    print(
        combined[
            ["topic", "topic_label"]
        ]
        .drop_duplicates()
        .sort_values("topic")
        .to_string(index=False)
    )

    return combined
