
"""
Automatic Weekly YouTube Trends Updater
Morocco Trends Project

Workflow:
YouTube trends
→ detect new videos
→ temporary audio
→ Whisper transcription
→ NLP
→ Gemini structured analysis
→ rebuild dashboard dataset
→ generate weekly creator report
"""

import os
import sys
from datetime import datetime

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from utils.config import DATA_DIR

from scripts.fetch_youtube import (
    connect_youtube,
    fetch_youtube_trends,
    create_metadata_dataframe,
)

from scripts.process_youtube_weekly import (
    process_new_youtube_videos
)

from scripts.analyze_weekly import (
    analyze_weekly_batch
)

from scripts.rebuild_dashboard import (
    rebuild_dashboard
)

from scripts.generate_weekly_report import (
    generate_weekly_report
)


def run_weekly_update():

    print("=" * 65)
    print("🇲🇦 MOROCCO YOUTUBE WEEKLY UPDATE")
    print("=" * 65)

    print(
        "Started:",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    # 1. Connect to YouTube
    print("\n[1/7] Connecting to YouTube...")

    youtube = connect_youtube()

    print("YouTube connected ✅")

    # 2. Fetch latest Morocco trends
    print("\n[2/7] Fetching current Morocco trends...")

    response = fetch_youtube_trends(
        youtube,
        region_code="MA",
        max_results=50
    )

    new_df = create_metadata_dataframe(
        response,
        DATA_DIR
    )

    # 3. No new videos
    if new_df is None or len(new_df) == 0:

        print("\nNo new trending videos this week ✅")

        print("\nRefreshing dashboard...")
        rebuild_dashboard()

        print("\nRefreshing weekly report...")
        generate_weekly_report()

        print("\nWeekly refresh completed ✅")

        return new_df

    print(
        f"\n🔥 {len(new_df)} new videos detected."
    )

    # 4. Transcription
    print("\n[3/7] Processing new videos...")

    processed_df = process_new_youtube_videos(
        new_df
    )

    usable_df = processed_df[
        processed_df["transcript"]
        .fillna("")
        .astype(str)
        .str.strip()
        != ""
    ].copy()

    print(
        f"\nUsable transcripts: "
        f"{len(usable_df)} / {len(processed_df)}"
    )

    if len(usable_df) == 0:

        print(
            "No usable transcripts produced. "
            "Stopping safely."
        )

        return processed_df

    # 5. NLP + Gemini
    print("\n[4/7] Running NLP + Gemini...")

    analyzed_df = analyze_weekly_batch(
        usable_df
    )

    # 6. Dashboard rebuild
    print("\n[5/7] Rebuilding dashboard dataset...")

    dashboard_df = rebuild_dashboard()

    # 7. Weekly creator report
    print("\n[6/7] Generating weekly trend report...")

    generate_weekly_report()

    print("\n[7/7] Weekly pipeline finished ✅")

    print(
        "Dashboard videos:",
        len(dashboard_df)
    )

    print("=" * 65)

    return analyzed_df


if __name__ == "__main__":
    run_weekly_update()
