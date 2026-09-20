
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

    # Keep all videos, including those analyzed using metadata only
    df["transcript"] = df["transcript"].fillna("").astype(str)

    # Mark videos with usable transcripts
    df["has_transcript"] = df["transcript"].str.strip().ne("")
    
    # Calculate word counts only for videos with transcripts
    df["transcript_word_count"] = pd.NA

    df.loc[
        df["has_transcript"],
        "transcript_word_count"
    ] = (
        df.loc[df["has_transcript"], "transcript"]
        .str.split()
        .str.len()
    )

    df["transcript_word_count"] = pd.to_numeric(
        df["transcript_word_count"],
        errors="coerce"
    )

    if os.path.exists(AI_PATH):
        ai_df = pd.read_csv(AI_PATH)
    else:
        ai_df = pd.DataFrame()

    return df, ai_df


df, ai_df = load_data()

# Videos with usable transcripts, for transcript-specific charts
transcript_df = df[df["has_transcript"]].copy()

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

st.header("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Analyzed Videos", len(df))
c2.metric("Total Views", f"{int(df['views'].sum()):,}")
c3.metric("Average Views", f"{int(df['views'].mean()):,}")
c4.metric("AI-Analyzed Videos", len(ai_df))

st.caption(
    f"This dashboard includes {len(df)} YouTube videos from the Morocco trends dataset: "
    f"{len(transcript_df)} with usable transcripts and "
    f"{len(df) - len(transcript_df)} without transcripts. "
    f"Gemini analysis is available for {len(ai_df)} videos, "
    "using transcripts where available and metadata only otherwise."
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
# 2. CONTENT CATEGORY PERFORMANCE
# =================================================

st.header("🎯 Which content categories perform best?")

if "content_category" in df.columns:

    category_perf = (
        df.groupby("content_category")
        .agg(
            videos=("video_id", "count"),
            average_views=("views", "mean"),
            median_views=("views", "median")
        )
        .reset_index()
        .sort_values("median_views", ascending=False)
    )

    fig_category = px.bar(
        category_perf,
        x="content_category",
        y="median_views",
        labels={
            "content_category": "Content Category",
            "median_views": "Median Views"
        }
    )

    st.plotly_chart(fig_category, use_container_width=True)

    display_category = category_perf.rename(
        columns={
            "content_category": "Content Category",
            "videos": "Videos",
            "average_views": "Average Views",
            "median_views": "Median Views"
        }
    )

    display_category["Average Views"] = (
        display_category["Average Views"].round(0).astype(int)
    )

    display_category["Median Views"] = (
        display_category["Median Views"].round(0).astype(int)
    )

    st.dataframe(
        display_category,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Videos are grouped into creator-friendly content categories based on "
        "their content, including Music, Gaming, Movies & Series, "
        "Entertainment & Challenges, and Sports & Football. Median views are "
        "used for comparison because breakout videos can strongly affect averages."
    )

else:
    st.warning(
        "Content categories are not available in the current dashboard dataset."
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

st.header("🗣️ Performance by detected transcript language")

if "language" in df.columns:

    language_names = {
        "ar": "Arabic / Darija",
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "lt": "Other / Uncertain",
        "no": "Other / Uncertain",
        "so": "Other / Uncertain"
    }

    language_perf = (
        transcript_df.groupby("language")
        .agg(
            videos=("video_id", "count"),
            average_views=("views", "mean"),
            median_views=("views", "median")
        )
        .reset_index()
    )

    language_perf["Language"] = (
        language_perf["language"]
        .map(language_names)
        .fillna("Other / Uncertain")
    )

    language_perf = (
        language_perf.groupby("Language")
        .agg(
            videos=("videos", "sum"),
            average_views=("average_views", "mean"),
            median_views=("median_views", "median")
    )
        .reset_index()
        .sort_values("median_views", ascending=False)
        )

    fig_language = px.bar(
        language_perf,
        x="Language",
        y="median_views",
        hover_data=["videos"],
        labels={
            "Language": "Detected Transcript Language",
            "median_views": "Median Views",
            "videos": "Number of Videos"
        }
     )

    st.plotly_chart(fig_language, use_container_width=True)

    display_language = language_perf[
        ["Language", "videos", "average_views", "median_views"]
    ].copy()

    display_language = display_language.rename(
        columns={
            "videos": "Videos",
            "average_views": "Average Views",
            "median_views": "Median Views"
        }
    )

    display_language["Average Views"] = (
        display_language["Average Views"].round(0).astype(int)
    )

    display_language["Median Views"] = (
        display_language["Median Views"].round(0).astype(int)
    )

    st.dataframe(
        display_language,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Language is automatically detected from video transcripts. "
        "Arabic detections may include Moroccan Darija. Results should be "
        "interpreted with sample size in mind: some language groups contain "
        "only one or two videos, and multilingual or noisy transcripts can "
        "produce uncertain language labels."
    )

else:
    st.info(
        "Detected transcript language is not available in the current dataset."
    )

st.divider()

# =================================================
# 5. TRANSCRIPT LENGTH VS VIEWS
# =================================================

st.header("⏱️ Does transcript length relate to video views?")

length_df = transcript_df[
    ["video_id", "title_caption", "transcript_word_count", "views"]
].dropna().copy()

length_df = length_df[
    (length_df["transcript_word_count"] >= 0)
    & (length_df["views"] >= 0)
]

if len(length_df) >= 2:

    # Original scatter plot: all videos
    st.subheader("All analyzed videos")

    fig_length = px.scatter(
        length_df,
        x="transcript_word_count",
        y="views",
        hover_name="title_caption",
        labels={
            "transcript_word_count": "Transcript Word Count",
            "views": "Views"
        }
    )

    st.plotly_chart(
        fig_length,
        use_container_width=True
    )

    # Correlation across the complete dataset
    correlation = length_df[
        "transcript_word_count"
    ].corr(length_df["views"])

    st.metric(
        "Transcript word count ↔ Views correlation",
        f"{correlation:.3f}"
    )

    st.caption(
        "The correlation measures the linear association between "
        "transcript word count and views in this dataset. "
        "A value close to zero indicates little linear association; "
        "it does not establish that transcript length has no effect "
        "on video performance."
    )

    # Additional chart for examining videos below the
    # 95th percentile of views
    view_limit = length_df["views"].quantile(0.95)

    zoom_df = length_df[
        length_df["views"] <= view_limit
    ].copy()

    st.subheader("Closer look at lower-view videos")

    fig_zoom = px.scatter(
        zoom_df,
        x="transcript_word_count",
        y="views",
        hover_name="title_caption",
        labels={
            "transcript_word_count": "Transcript Word Count",
            "views": "Views"
        }
    )

    st.plotly_chart(
        fig_zoom,
        use_container_width=True
    )

    st.caption(
        f"This second chart displays {len(zoom_df)} of "
        f"{len(length_df)} videos, excluding videos above the "
        "95th percentile of views to make the remaining "
        "points easier to examine. The correlation shown above "
        "is calculated using the full dataset."
    )

    st.info(
        "Transcript word count measures the amount of transcribed "
        "speech or lyrics, not the actual duration of a video. "
        "Music, gaming, and other content formats can have very "
        "different amounts of speech."
    )

else:
    st.info(
        "Not enough valid data to analyze transcript length "
        "and video views."
    )

st.divider()

# =================================================
# 6. HASHTAG TRENDS
# =================================================

st.header("🏷️ Which hashtags appear most often?")

if "hashtags" in df.columns:

    hashtag_counts = Counter()
    hashtag_pairs = []

    videos_with_hashtags = 0

    for tags in df["hashtags"].dropna():

        clean_tags = sorted(
            set(
                tag.strip().lower()
                for tag in str(tags).split(",")
                if tag.strip()
            )
        )

        if not clean_tags:
            continue

        videos_with_hashtags += 1

        # Count each hashtag only once per video
        hashtag_counts.update(clean_tags)

        # Count hashtag combinations within each video
        hashtag_pairs.extend(
            combinations(clean_tags, 2)
        )

    # ---------------------------------------------
    # Individual hashtag frequency
    # ---------------------------------------------

    top_hashtags = pd.DataFrame(
        [
            {
                "Hashtag": tag,
                "Videos": count
            }
            for tag, count in hashtag_counts.most_common(10)
        ]
    )

    if not top_hashtags.empty:

        st.subheader("Most frequent individual hashtags")

        fig_individual = px.bar(
            top_hashtags.sort_values(
                "Videos",
                ascending=True
            ),
            x="Videos",
            y="Hashtag",
            orientation="h",
            labels={
                "Videos": "Number of Videos",
                "Hashtag": "Hashtag"
            }
        )

        fig_individual.update_layout(
            height=500,
            yaxis_title=None
        )

        st.plotly_chart(
            fig_individual,
            use_container_width=True
        )

        st.caption(
            f"{videos_with_hashtags} of {len(df)} analyzed videos "
            "contain hashtags. Each hashtag is counted at most "
            "once per video."
        )

    else:

        st.info(
            "No usable hashtags were found in this dataset."
        )

    # ---------------------------------------------
    # Hashtag co-occurrence
    # ---------------------------------------------

    st.subheader("Which hashtags appear together?")

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
            orientation="h",
            labels={
                "Co-occurrences": "Number of Videos",
                "Pair": "Hashtag Pair"
            }
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
            "Each pair is counted once per video. Multiple "
            "pairs may come from the same video, so these "
            "counts should not be interpreted as independent "
            "trends. Frequent appearance does not establish "
            "that a hashtag increases views."
        )

    else:

        st.info(
            "Not enough hashtag combinations were found "
            "for co-occurrence analysis."
        )

else:

    st.info(
        "Hashtag data is not available."
    )

st.divider()

# =================================================
# 7. GEMINI THEMES
# =================================================

st.header("🤖 What is Gemini detecting in the content?")

if not ai_df.empty:

    transcript_ai_count = (
        ai_df["analysis_source"].eq("transcript").sum()
        if "analysis_source" in ai_df.columns else len(ai_df)
    )

    metadata_ai_count = (
        ai_df["analysis_source"].eq("metadata_only").sum()
        if "analysis_source" in ai_df.columns else 0
    )

    st.info(
        f"Gemini analysis is available for {len(ai_df)} of {len(df)} videos: "
        f"{transcript_ai_count} transcript-based and "
        f"{metadata_ai_count} metadata-only."
    )

    columns_to_show = [
        col for col in [
            "title_caption",
            "analysis_source",
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
        f"Gemini analysis is available for {len(ai_df)} of {len(df)} videos. "
        f"{transcript_ai_count} analyses use transcripts, while "
        f"{metadata_ai_count} use metadata only. "
        "Metadata-only results are limited to information supported by "
        "video titles and available metadata; they do not establish "
        "what was said or shown in the videos."
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
- Videos in the dashboard: **{len(df)}**
- Videos with usable transcripts: **{len(transcript_df)}**
- Videos without transcripts: **{len(df) - len(transcript_df)}**
- Gemini transcript-based analyses: **{transcript_ai_count}**
- Gemini metadata-only analyses: **{metadata_ai_count}**
- Total Gemini analyses: **{len(ai_df)} / {len(df)}**
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
