
import os
from itertools import combinations
from collections import Counter

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Morocco YouTube Trends",
    page_icon="🇲🇦",
    layout="wide"
)

DATA_PATH = "data/dashboard_data.csv"
AI_PATH = "data/gemini_analysis.csv"

# =================================================
# LOAD DATA
# =================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    # Keep only usable transcripts
    df = df[df["transcript"].notna()].copy()
    df = df[
        df["transcript"].astype(str).str.strip() != ""
    ].reset_index(drop=True)

    # Create transcript word count if needed
    if "transcript_word_count" not in df.columns:
        df["transcript_word_count"] = (
            df["transcript"]
            .astype(str)
            .str.split()
            .str.len()
        )

    if os.path.exists(AI_PATH):
        ai_df = pd.read_csv(AI_PATH)
    else:
        ai_df = pd.DataFrame()

    return df, ai_df


df, ai_df = load_data()

# =================================================
# HEADER
# =================================================

st.title("🇲🇦 Morocco YouTube Trends")
st.subheader("Creator-focused trend analysis")

st.write(
    "A dashboard designed to help creators understand what is gaining "
    "attention on YouTube in Morocco, how audiences respond, and what "
    "content signals are appearing in the current dataset."
)

if st.button("🔄 Refresh Dashboard"):
    st.cache_data.clear()
    st.rerun()

st.divider()

# =================================================
# OVERVIEW
# =================================================

st.header("📊 Current Snapshot")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Usable Videos", len(df))
c2.metric("Total Views", f"{int(df['views'].sum()):,}")
c3.metric("Average Views", f"{int(df['views'].mean()):,}")
c4.metric("Gemini Analyses", f"{len(ai_df)} / {len(df)}")

st.caption(
    f"The quantitative analysis uses {len(df)} videos with usable transcripts. "
    f"Gemini 2.5 Flash theme extraction is currently available for "
    f"{len(ai_df)} videos."
)

if len(ai_df) < len(df):
    st.warning(
        "Gemini analysis is incomplete because API/runtime limits interrupted "
        "processing. No missing AI results have been fabricated."
    )

st.divider()

# =================================================
# 1. TOP CONTENT
# =================================================

st.header("🔥 What content is attracting the most views?")

top_videos = (
    df.nlargest(10, "views")[
        ["title_caption", "views"]
    ]
    .sort_values("views", ascending=True)
)

fig_top = px.bar(
    top_videos,
    x="views",
    y="title_caption",
    orientation="h",
    labels={
        "views": "Views",
        "title_caption": "Video"
    }
)

fig_top.update_layout(
    height=550,
    yaxis_title=None
)

st.plotly_chart(fig_top, use_container_width=True)

st.caption(
    "The distribution contains major breakout videos, so the highest-view "
    "content can strongly influence averages."
)

st.divider()

# =================================================
# 2. TOPIC PERFORMANCE
# =================================================

st.header("🎯 Which topics perform best?")

if "topic" in df.columns:

    topic_perf = (
        df.groupby("topic")
        .agg(
            videos=("video_id", "count"),
            average_views=("views", "mean"),
            median_views=("views", "median")
        )
        .reset_index()
        .sort_values("median_views", ascending=False)
    )

    topic_perf["topic"] = topic_perf["topic"].astype(int)

    # Use dynamic weekly topic descriptions when available
    if "topic_label" in df.columns:
        label_map = (
            df[["topic", "topic_label"]]
            .drop_duplicates(subset=["topic"])
            .set_index("topic")["topic_label"]
            .to_dict()
        )

        topic_perf["Topic"] = (
            topic_perf["topic"]
            .map(label_map)
            .fillna(
                topic_perf["topic"].apply(
                    lambda x: f"Topic {x}"
                )
            )
        )
    else:
        topic_perf["Topic"] = (
            topic_perf["topic"]
            .apply(lambda x: f"Topic {x}")
        )

    fig_topic = px.bar(
        topic_perf,
        x="Topic",
        y="median_views",
        labels={
            "Topic": "Topic Cluster",
            "median_views": "Median Views"
        }
    )

    st.plotly_chart(fig_topic, use_container_width=True)

    st.dataframe(
        topic_perf,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The six topic clusters come from the Phase 1 text analysis. "
        "Some clusters are influenced by language differences, so they "
        "should not be treated as perfect semantic categories."
    )

else:
    st.info(
        "Topic cluster labels are not stored in the current dashboard dataset."
    )

st.divider()

# =================================================
# 3. ENGAGEMENT
# =================================================

st.header("❤️ How are views, likes and comments related?")

engagement_corr = (
    df[["views", "likes", "comments"]]
    .corr()
    .round(3)
)

st.dataframe(
    engagement_corr,
    use_container_width=True
)

e1, e2, e3 = st.columns(3)

e1.metric(
    "Views ↔ Likes",
    f"{df['views'].corr(df['likes']):.3f}"
)

e2.metric(
    "Views ↔ Comments",
    f"{df['views'].corr(df['comments']):.3f}"
)

e3.metric(
    "Likes ↔ Comments",
    f"{df['likes'].corr(df['comments']):.3f}"
)

st.caption(
    "Views, likes and comments move strongly together in this dataset. "
    "These relationships are correlations and do not prove causation."
)

st.divider()

# =================================================
# 4. LANGUAGE
# =================================================

st.header("🗣️ What language does the audience respond to?")

if "language" in df.columns:

    language_perf = (
        df.groupby("language")
        .agg(
            videos=("video_id", "count"),
            average_views=("views", "mean"),
            median_views=("views", "median")
        )
        .reset_index()
        .sort_values("median_views", ascending=False)
    )

    fig_language = px.bar(
        language_perf,
        x="language",
        y="median_views",
        labels={
            "language": "Detected Language",
            "median_views": "Median Views"
        }
    )

    st.plotly_chart(fig_language, use_container_width=True)

    st.dataframe(
        language_perf,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Language groups are uneven. English results are influenced by a "
        "major high-view video, while some one-video language labels may "
        "reflect multilingual or noisy speech-to-text transcripts. "
        "Arabic detection may also include Darija."
    )

else:
    st.info("Language labels are not available in the current dataset.")

st.divider()

# =================================================
# 5. CONTENT LENGTH
# =================================================

st.header("⏱️ Does longer spoken content get more views?")

scatter_df = df[
    ["transcript_word_count", "views", "title_caption"]
].copy()

fig_length = px.scatter(
    scatter_df,
    x="transcript_word_count",
    y="views",
    hover_name="title_caption",
    labels={
        "transcript_word_count": "Transcript Word Count",
        "views": "Views"
    }
)

st.plotly_chart(fig_length, use_container_width=True)

length_corr = df[
    "transcript_word_count"
].corr(df["views"])

st.metric(
    "Transcript length ↔ Views correlation",
    f"{length_corr:.3f}"
)

st.caption(
    "The correlation is close to zero, meaning transcript length has "
    "almost no linear relationship with views in this dataset."
)

st.divider()

# =================================================
# 6. HASHTAG TRENDS
# =================================================

st.header("🏷️ Which hashtags repeatedly appear together?")

if "hashtags" in df.columns:

    hashtag_pairs = []

    for tags in df["hashtags"].dropna():

        clean_tags = sorted(
            set(
                tag.strip().lower()
                for tag in str(tags).split(",")
                if tag.strip()
            )
        )

        hashtag_pairs.extend(
            combinations(clean_tags, 2)
        )

    pair_counts = Counter(hashtag_pairs)

    top_pairs = pd.DataFrame(
        [
            {
                "Hashtag 1": pair[0],
                "Hashtag 2": pair[1],
                "Co-occurrences": count
            }
            for pair, count in pair_counts.most_common(10)
        ]
    )

    if not top_pairs.empty:

        top_pairs["Pair"] = (
            top_pairs["Hashtag 1"]
            + " + "
            + top_pairs["Hashtag 2"]
        )

        fig_tags = px.bar(
            top_pairs.sort_values(
                "Co-occurrences",
                ascending=True
            ),
            x="Co-occurrences",
            y="Pair",
            orientation="h"
        )

        fig_tags.update_layout(
            height=500,
            yaxis_title=None
        )

        st.plotly_chart(
            fig_tags,
            use_container_width=True
        )

        st.caption(
            "The strongest recurring hashtag combinations are concentrated "
            "around Free Fire and related gaming content. Counts are small, "
            "so these should be read as signals rather than broad conclusions."
        )

    else:
        st.info(
            "There are not enough repeated hashtags for co-occurrence analysis."
        )

st.divider()

# =================================================
# 7. GEMINI THEMES
# =================================================

st.header("🤖 What is Gemini detecting in the content?")

if not ai_df.empty:

    st.info(
        f"Gemini 2.5 Flash successfully extracted themes from "
        f"{len(ai_df)} of {len(df)} usable videos."
    )

    columns_to_show = [
        col for col in [
            "title_caption",
            "main_topic",
            "emotional_tone",
            "tone_shift",
            "language_register",
            "intended_audience",
            "trend_or_cultural_moment",
            "creator_angle",
            "shareability_reason"
        ]
        if col in ai_df.columns
    ]

    st.dataframe(
        ai_df[columns_to_show],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "This is a partial Gemini sample because processing was interrupted "
        "by the API/runtime limitation. It should not be presented as a "
        "complete 46-video AI trend report."
    )

else:
    st.warning("No Gemini analysis results are available.")

st.divider()

# =================================================
# 8. CREATOR SIGNALS
# =================================================

st.header("💡 What should a creator take from this?")

st.markdown("""
**Current signals from the quantitative analysis:**

- Breakout videos can dominate overall performance, so averages alone can be misleading.
- Views, likes and comments are strongly related to each other.
- Transcript length does not appear to explain whether a video receives more views.
- Gaming-related hashtag combinations, especially Free Fire-related tags, are the strongest recurring hashtag signal in this dataset.
- Language performance varies, but unequal group sizes and noisy multilingual transcripts mean the comparison should be interpreted cautiously.
- Topic clusters provide useful patterns, but some are partly separated by language rather than topic alone.
""")

st.divider()

# =================================================
# 9. WHAT TO MAKE THIS WEEK
# =================================================

st.header("📌 What could a creator make this week?")

st.write(
    "Based on the current Phase 1 evidence, gaming content is one of the "
    "clearest recurring signals because Free Fire-related hashtags repeatedly "
    "appear together. Creators should focus less on simply making longer "
    "videos and more on the topic, audience relevance, and engagement potential."
)

st.write(
    "The Gemini results can add information about emotional tone, audience, "
    "creator angle and shareability, but the final AI-based weekly trend report "
    "should only be treated as complete once all usable videos have been "
    "processed."
)

st.divider()

# =================================================
# LIMITATIONS
# =================================================

with st.expander("⚠️ Data and Analysis Limitations"):

    st.markdown(f"""
- Original dataset: **49 videos**
- Videos with usable transcripts: **{len(df)}**
- Videos excluded from text analysis because transcripts were empty: **3**
- Gemini analyses currently completed: **{len(ai_df)} / {len(df)}**
- Speech-to-text data is multilingual and sometimes noisy.
- Some automatic language labels may be inaccurate for multilingual content.
- Topic clusters can be influenced by language.
- A major high-view outlier affects some averages.
- Correlation does not imply causation.
""")

st.caption(
    "Morocco YouTube Trends — Internship Phase 2 creator dashboard"
)


# =================================================
# WEEKLY AI TREND REPORT
# =================================================

st.divider()

st.header("📝 Weekly Trend Report")

REPORT_PATH = "data/weekly_trend_report.md"

if os.path.exists(REPORT_PATH):

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        weekly_report = f.read()

    st.markdown(weekly_report)

else:

    st.info(
        "The weekly AI trend report will appear here "
        "after the first automated weekly update."
    )
