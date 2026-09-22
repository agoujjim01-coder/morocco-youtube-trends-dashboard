
import os
import json
import pandas as pd
import google.generativeai as genai

from utils.config import DATA_DIR


def generate_weekly_report():
    """
    Generate a creator-friendly weekly YouTube trend report.

    Uses:
    - weekly_analyzed_youtube.csv
    - gemini_analysis.csv

    Saves:
    - weekly_trend_report.md
    """

    weekly_path = os.path.join(
        DATA_DIR,
        "weekly_analyzed_youtube.csv"
    )

    ai_path = os.path.join(
        DATA_DIR,
        "gemini_analysis.csv"
    )

    report_path = os.path.join(
        DATA_DIR,
        "weekly_trend_report.md"
    )

    # -------------------------------------------------
    # Load weekly data
    # -------------------------------------------------

    if not os.path.exists(weekly_path):

        print("No weekly analysed dataset found.")
        return None

    weekly_df = pd.read_csv(
        weekly_path
    )

    if len(weekly_df) == 0:

        print("Weekly dataset is empty.")
        return None

    # -------------------------------------------------
    # Basic weekly statistics
    # -------------------------------------------------

    total_videos = len(weekly_df)

    avg_views = int(
        weekly_df["views"].mean()
    )

    median_views = int(
        weekly_df["views"].median()
    )

    top_videos = (
        weekly_df
        .nlargest(5, "views")[
            [
                "title_caption",
                "views",
                "likes",
                "comments",
                "language"
            ]
        ]
        .to_dict("records")
    )

    language_summary = (
        weekly_df["language"]
        .value_counts()
        .to_dict()
    )

    # -------------------------------------------------
    # Load Gemini results for this week's IDs
    # -------------------------------------------------

    weekly_ids = set(
        weekly_df["video_id"]
        .astype(str)
    )

    ai_records = []

    if os.path.exists(ai_path):

        ai_df = pd.read_csv(
            ai_path
        )

        if "video_id" in ai_df.columns:

            weekly_ai = ai_df[
                ai_df["video_id"]
                .astype(str)
                .isin(weekly_ids)
            ].copy()

            useful_columns = [
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

            useful_columns = [
                col
                for col in useful_columns
                if col in weekly_ai.columns
            ]

            ai_records = (
                weekly_ai[
                    useful_columns
                ]
                .fillna("unclear")
                .to_dict("records")
            )

    # -------------------------------------------------
    # Gemini API
    # -------------------------------------------------

    api_key = os.environ.get(
        "GEMINI_API_KEY"
    )

    if not api_key:

        print(
            "GEMINI_API_KEY not available. "
            "Creating fallback report."
        )

        fallback = f"""
# Morocco YouTube Weekly Trend Report

## Weekly Snapshot

- Videos analysed: {total_videos}
- Average views: {avg_views:,}
- Median views: {median_views:,}

## Language Distribution

{json.dumps(language_summary, ensure_ascii=False, indent=2)}

## Top Performing Videos

{json.dumps(top_videos, ensure_ascii=False, indent=2)}

## Note

The AI narrative report could not be generated because the Gemini API key was unavailable. The dashboard quantitative data was still updated successfully.
"""

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(fallback.strip())

        print("Fallback weekly report saved ✅")

        return fallback

    genai.configure(
        api_key=api_key
    )

    model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

    # -------------------------------------------------
    # Report prompt
    # -------------------------------------------------

    prompt = f"""
You are writing a weekly YouTube trend report for Moroccan content creators.

Your job is NOT to sound impressive.
Your job is to give practical, evidence-based creator insights.

Use ONLY the dataset supplied below.
Do not invent information.
Do not claim causation.
If evidence is weak, say so clearly.

WEEKLY DATASET SUMMARY

Videos analysed:
{total_videos}

Average views:
{avg_views}

Median views:
{median_views}

Language distribution:
{json.dumps(language_summary, ensure_ascii=False)}

Top videos:
{json.dumps(top_videos, ensure_ascii=False)}

Structured Gemini analyses:
{json.dumps(ai_records, ensure_ascii=False)}

DATA QUALITY RULES:
- Some videos may have no transcript and may be analysed using metadata only.
- "unknown" language means spoken language was not established.
- "unclear" emotional tone or register must not be presented as an observed characteristic.
- Do not describe what was said or shown in metadata-only videos.
- Distinguish observed engagement counts from explanations for engagement.
- A title may suggest a topic but does not verify the video's contents.

Write a concise weekly report using exactly these sections:

# Morocco YouTube Weekly Trend Report

## 1. What is dominating this week?
Explain the strongest topics or content types.

## 2. What language and tone are audiences seeing?
Describe language mix, tone, and register without overclaiming.

## 3. What is driving engagement?
Explain patterns that may help explain why videos are attracting attention.

## 4. Content opportunities
Identify gaps or opportunities supported by the available evidence.

## 5. What should a Moroccan creator make this week?
Give 2 to 3 concrete content recommendations.
For each recommendation include:
- topic or format
- suggested language/register
- creator angle
- why it is relevant this week

## 6. Data caution
Briefly explain limitations in the weekly sample.

Write for a creator, not a data scientist.
Keep it practical and readable.
"""

    try:

        response = model.generate_content(
            prompt,
            request_options={"timeout": 30}
        )
        
        report = response.text.strip()

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(report)

        print("AI weekly trend report generated ✅")
        print("Saved to:", report_path)

        return report

    except Exception as e:

        print("Gemini weekly report failed ❌")
        print(e)

        # ---------------------------------------------
        # Safe fallback
        # ---------------------------------------------

        fallback = f"""
# Morocco YouTube Weekly Trend Report

## Weekly Snapshot

- Videos analysed: {total_videos}
- Average views: {avg_views:,}
- Median views: {median_views:,}

## Top Performing Videos

{json.dumps(top_videos, ensure_ascii=False, indent=2)}

## Language Distribution

{json.dumps(language_summary, ensure_ascii=False, indent=2)}

## Data caution

The AI narrative layer was unavailable during this update. Quantitative dashboard results were still refreshed successfully.
"""

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(fallback.strip())

        print("Fallback report saved ✅")

        return fallback
