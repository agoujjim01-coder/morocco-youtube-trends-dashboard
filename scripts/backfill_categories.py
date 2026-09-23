import os
import json
import time
import pandas as pd
import google.generativeai as genai

from utils.config import DATA_DIR

from scripts.analyze_weekly import (
    build_gemini_prompt,
    build_metadata_only_prompt
)


ALLOWED_CATEGORIES = {
    "Music",
    "Gaming",
    "Movies & Series",
    "Entertainment & Challenges",
    "Sports & Football",
    "Education & Tutorials",
    "News & Current Events",
    "Lifestyle & Vlogs",
    "Food & Cooking",
    "Technology",
    "Comedy",
    "Other / Unclear"
}

MAX_VIDEOS_PER_RUN = 10


def backfill_categories():

    dashboard_path = os.path.join(
        DATA_DIR,
        "dashboard_data.csv"
    )

    ai_path = os.path.join(
        DATA_DIR,
        "gemini_analysis.csv"
    )

    if not os.path.exists(dashboard_path):
        print("Dashboard dataset not found.")
        return

    if not os.path.exists(ai_path):
        print("Gemini analysis history not found.")
        return

    dashboard_df = pd.read_csv(dashboard_path)
    ai_df = pd.read_csv(ai_path)

    if "content_category" not in ai_df.columns:
        ai_df["content_category"] = pd.NA

    dashboard_df["video_id"] = (
        dashboard_df["video_id"].astype(str)
    )

    ai_df["video_id"] = (
        ai_df["video_id"].astype(str)
    )

    # Only classify videos without a saved category.
    missing_category = (
        ai_df["content_category"].isna()
        | ai_df["content_category"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    pending_ids = set(
        ai_df.loc[
            missing_category,
            "video_id"
        ]
    )

    pending_df = dashboard_df[
        dashboard_df["video_id"].isin(pending_ids)
    ].copy()

    print(
        "Videos needing categories:",
        len(pending_df)
    )

    if pending_df.empty:
        print("All existing videos have categories ✅")
        return

    api_key = os.environ.get(
        "GEMINI_API_KEY", ""
    ).strip()

    if not api_key:
        print("GEMINI_API_KEY is missing.")
        return

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

    batch = pending_df.head(
        MAX_VIDEOS_PER_RUN
    )

    successful = 0

    for _, row in batch.iterrows():

        video_id = str(row["video_id"])
        title = str(row["title_caption"])

        transcript_value = row.get(
            "transcript", ""
        )

        transcript = (
            ""
            if pd.isna(transcript_value)
            else str(transcript_value).strip()
        )

        print(
            f"\nClassifying {video_id}: "
            f"{title[:60]}",
            flush=True
        )

        if transcript:

            prompt = build_gemini_prompt(
                title,
                transcript
            )

        else:

            prompt = build_metadata_only_prompt(
                title,
                row.get("views", "unknown"),
                row.get("likes", "unknown"),
                row.get("comments", "unknown")
            )

        try:

            response = model.generate_content(
                prompt,
                request_options={"timeout": 30}
            )

            response_text = (
                response.text
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            result = json.loads(
                response_text
            )

            category = result.get(
                "content_category",
                "Other / Unclear"
            )

            if category not in ALLOWED_CATEGORIES:
                category = "Other / Unclear"

            ai_df.loc[
                ai_df["video_id"] == video_id,
                "content_category"
            ] = category

            # Save after every successful classification
            # so progress is not lost if a later call fails.
            ai_df.to_csv(
                ai_path,
                index=False
            )

            successful += 1

            print(
                f"Category saved: {category} ✅",
                flush=True
            )

            time.sleep(3)

        except Exception as e:

            print(
                f"Classification failed for {video_id}:",
                flush=True
            )

            print(e, flush=True)

            # Leave the category missing so a later
            # run can retry this video.

    print("\n" + "=" * 50)
    print("CATEGORY BACKFILL SUMMARY")
    print("=" * 50)

    print(
        "Successfully classified:",
        successful
    )

    remaining = (
        ai_df["content_category"].isna()
        | ai_df["content_category"]
        .astype(str)
        .str.strip()
        .eq("")
    ).sum()

    print(
        "Remaining without categories:",
        remaining
    )


if __name__ == "__main__":
    backfill_categories()
  
