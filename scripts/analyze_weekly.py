
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

Return ONLY valid JSON using exactly this structure:

{{
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
        .apply(detect_language)
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

    api_key = os.environ.get("GEMINI_API_KEY")

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

            transcript = str(
                row.get("transcript", "")
            ).strip()

            if not transcript:

                print(
                    f"Skipping Gemini: "
                    f"{row['video_id']} "
                    f"(empty transcript)"
                )

                continue

            print(
                f"Gemini: "
                f"{row['video_id']} "
                f"- {row['title_caption'][:50]}"
            )

            prompt = build_gemini_prompt(
                row["title_caption"],
                transcript
            )

            try:

                response = model.generate_content(
                    prompt
                )

                text = (
                    response.text
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

                result = json.loads(text)

                result["video_id"] = row["video_id"]
                result["title_caption"] = row["title_caption"]

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

                # Continue processing other videos
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

        ai_combined = (
            ai_combined
            .drop_duplicates(
                subset=["video_id"],
                keep="last"
            )
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
