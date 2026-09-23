
import os
import re
import json
import time
import pandas as pd

from langdetect import detect
import google.generativeai as genai

from utils.config import DATA_DIR


# -------------------------------------------------
# NLP helpers
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


# -------------------------------------------------
# Gemini prompt
# -------------------------------------------------

def build_gemini_prompt(title, transcript):

    return f"""
You are analysing a YouTube video transcript for creator trend analysis.

IMPORTANT RULES:
- Use ONLY information explicitly supported by the VIDEO TITLE and TRANSCRIPT below.
- Do NOT use outside knowledge about the creator, artist, song, brand, genre, popularity, or audience.
- Do NOT assume fame, fanbase size, cultural importance, or genre unless clearly supported.
- The transcript may contain Moroccan Darija, Arabic, French, English, other languages, or speech-to-text errors.
- If the transcript is too noisy to support a conclusion, return "unclear".
- Prefer "unclear" over guessing.

CONTENT CATEGORY RULES:
- Assign exactly ONE content_category from this list:
  Music
  Gaming
  Movies & Series
  Entertainment & Challenges
  Sports & Football
  Education & Tutorials
  News & Current Events
  Lifestyle & Vlogs
  Food & Cooking
  Technology
  Comedy
  Other / Unclear
- Use ONLY the video title and transcript.
- Do not infer the category from outside knowledge.
- If there is insufficient evidence, use "Other / Unclear".

Return ONLY valid JSON using exactly this structure:

{{
  "content_category": "",
  "main_topic": "",
  "emotional_tone": "",
  "tone_shift": "",
  "language_register": "",
  "intended_audience": "",
  "trend_or_cultural_moment": "",
  "creator_angle": "",
  "shareability_reason": ""
}}

VIDEO TITLE:
{title}

TRANSCRIPT:
{transcript}
"""


# -------------------------------------------------
# Main analysis
# -------------------------------------------------

def build_metadata_only_prompt(title, views, likes, comments):

    return f"""
You are analysing a YouTube video's available metadata
for creator trend analysis.

IMPORTANT RULES:
- No transcript or video content is available.
- Use ONLY the title and numerical metadata below.
- Do NOT claim to know what was said or shown in the video.
- Do NOT infer spoken language from the title.
- Do NOT invent emotional tone, tone shifts, or language register.
- Use "unclear" when the metadata does not support an answer.
- Do NOT use outside knowledge about the creator or video.
- A title can suggest a topic, but does not verify the video's content.

CONTENT CATEGORY RULES:
- Assign exactly ONE content_category from this list:
  Music
  Gaming
  Movies & Series
  Entertainment & Challenges
  Sports & Football
  Education & Tutorials
  News & Current Events
  Lifestyle & Vlogs
  Food & Cooking
  Technology
  Comedy
  Other / Unclear
- Classify using ONLY the video title and available metadata.
- Do not assume what happens inside the video.
- If the title does not provide enough evidence, use "Other / Unclear".

Return ONLY valid JSON using exactly this structure:

{{
  "content_category": "", 
  "main_topic": "",
  "emotional_tone": "unclear",
  "tone_shift": "unclear",
  "language_register": "unclear",
  "intended_audience": "",
  "trend_or_cultural_moment": "",
  "creator_angle": "",
  "shareability_reason": ""
}}

VIDEO TITLE:
{title}

VIEWS:
{views}

LIKES:
{likes}

COMMENTS:
{comments}
"""

def analyze_weekly_batch(df):

    if df is None or len(df) == 0:

        print("No weekly videos to analyse ✅")

        return df

    df = df.copy()

    # ---------------------------------------------
    # NLP
    # ---------------------------------------------

    print("\nRunning NLP...")

    df["clean_transcript"] = (
        df["transcript"]
        .fillna("")
        .apply(clean_text)
    )

    df["language"] = (
        df["clean_transcript"]
        .apply(
            lambda text: detect_language(text)
            if str(text).strip()
            else "unknown"
        )
    )

    df["transcript_word_count"] = (
        df["transcript"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    print("NLP completed ✅")

    # ---------------------------------------------
    # Gemini
    # ---------------------------------------------

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    
    if not api_key:

        try:
            from google.colab import userdata

            # Change this ONLY if your Gemini
            # Colab secret has a different name.
            api_key = userdata.get("GEMINI_API_KEY")

        except Exception:
            api_key = None

    ai_results = []

    if not api_key:

        print(
            "Gemini API key not available. "
            "NLP will continue without Gemini."
        )

    else:

        genai.configure(
            api_key=api_key
        )

        model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

        print("\nRunning Gemini analysis...")

        for i, row in df.iterrows():

            transcript_value = row.get("transcript", "")

            transcript = (
                ""
                if pd.isna(transcript_value)
                else str(transcript_value).strip()
            )

            analysis_source = (
                "transcript"
                if transcript
                else "metadata_only"
            )

            print(
                f"Gemini: "
                f"{row['video_id']} "
                f"- {row['title_caption'][:50]}"
            )

            if analysis_source == "transcript":

                prompt = build_gemini_prompt(
                    row["title_caption"],
                    transcript
                )

            else:

                prompt = build_metadata_only_prompt(
                    row["title_caption"],
                    row.get("views", "unknown"),
                    row.get("likes", "unknown"),
                    row.get("comments", "unknown")
                )

            try:

                response = model.generate_content(
                    prompt,
                    request_options={"timeout": 30}
                )

                text = (
                    response.text
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

                result = json.loads(text)
                
                if analysis_source == "metadata_only":
                    result["emotional_tone"] = "unclear"
                    result["tone_shift"] = "unclear"
                    result["language_register"] = "unclear"

                result["video_id"] = row["video_id"]
                result["title_caption"] = row["title_caption"]
                result["analysis_source"] = analysis_source
                
                ai_results.append(
                    result
                )

                print("Gemini saved ✅")

                # Avoid hammering the API
                time.sleep(8)

            except Exception as e:

                print(
                    f"Gemini failed for "
                    f"{row['video_id']}:"
                )
            
                print(e)
            
                ai_results.append({
                    "video_id": row["video_id"],
                    "title_caption": row["title_caption"],
                    "content_category": "Other / Unclear",
                    "analysis_source": analysis_source,
                    "main_topic": "unclear",
                    "emotional_tone": "unclear",
                    "tone_shift": "unclear",
                    "language_register": "unclear",
                    "intended_audience": "unclear",
                    "trend_or_cultural_moment": "unclear",
                    "creator_angle": "unclear",
                    "shareability_reason": "unclear"
                })
            
                time.sleep(15)

    # ---------------------------------------------
    # Save Gemini history
    # ---------------------------------------------

    if ai_results:

        ai_new = pd.DataFrame(
            ai_results
        )

        ai_path = os.path.join(
            DATA_DIR,
            "gemini_analysis.csv"
        )

        if os.path.exists(ai_path):

            old_ai = pd.read_csv(
                ai_path
            )

            ai_combined = pd.concat(
                [old_ai, ai_new],
                ignore_index=True
            )

        else:

            ai_combined = ai_new

        # Prefer transcript-based analysis for repeated videos.
        # If both analyses have the same source,
        # keep the newer one.

        ai_combined["_source_priority"] = (
            ai_combined["analysis_source"]
            .fillna("")
            .eq("transcript")
            .astype(int)
        )

        ai_combined = (
            ai_combined
            .sort_values(
                "_source_priority",
                kind="stable"
            )
            .drop_duplicates(
                subset=["video_id"],
                keep="last"
            )
            .drop(columns=["_source_priority"])
            .reset_index(drop=True)
        )

        ai_combined.to_csv(
            ai_path,
            index=False
        )

        print(
            "\nGemini history updated ✅"
        )

        print(
            "Total AI analyses:",
            len(ai_combined)
        )

    # ---------------------------------------------
    # Save analysed weekly batch
    # ---------------------------------------------

    weekly_path = os.path.join(
        DATA_DIR,
        "weekly_analyzed_youtube.csv"
    )

    df.to_csv(
        weekly_path,
        index=False
    )

    print(
        "\nWeekly analysed dataset saved ✅"
    )

    return df
