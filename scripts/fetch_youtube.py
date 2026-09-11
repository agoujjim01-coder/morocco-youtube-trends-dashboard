
import os
import re
import pandas as pd
from datetime import datetime, timezone
from googleapiclient.discovery import build


def connect_youtube():
    """
    Connect to YouTube Data API v3.

    GitHub Actions:
        YOUTUBE_API_KEY environment variable

    Colab testing:
        Morocco_Trends_API secret
    """

    api_key = os.environ.get("YOUTUBE_API_KEY")

    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("Morocco_Trends_API")
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError(
            "YouTube API key not found."
        )

    return build(
        "youtube",
        "v3",
        developerKey=api_key
    )


def fetch_youtube_trends(
    youtube,
    region_code="MA",
    max_results=50
):
    """
    Fetch current YouTube most-popular videos
    for Morocco.
    """

    return (
        youtube.videos()
        .list(
            part="snippet,statistics",
            chart="mostPopular",
            regionCode=region_code,
            maxResults=max_results
        )
        .execute()
    )


def _extract_hashtags(snippet):

    text = (
        str(snippet.get("title", ""))
        + " "
        + str(snippet.get("description", ""))
    )

    hashtags = re.findall(
        r"#([^\s#]+)",
        text
    )

    hashtags = list(
        dict.fromkeys(hashtags)
    )

    return ", ".join(hashtags)


def create_metadata_dataframe(
    response,
    data_dir
):
    """
    Build the current Morocco YouTube snapshot.

    IMPORTANT:
    A video is considered already processed ONLY
    if its ID exists in dashboard_data.csv.

    trending_metadata.csv is metadata history.
    It does NOT prevent unfinished videos from
    being processed on the next run.
    """

    os.makedirs(
        data_dir,
        exist_ok=True
    )

    scraped_at = datetime.now(
        timezone.utc
    ).isoformat()

    rows = []

    for item in response.get("items", []):

        snippet = item.get(
            "snippet",
            {}
        )

        stats = item.get(
            "statistics",
            {}
        )

        video_id = str(
            item.get("id", "")
        )

        rows.append({
            "platform": "youtube",
            "video_id": video_id,
            "url":
                f"https://www.youtube.com/watch?v={video_id}",
            "title_caption":
                snippet.get("title", ""),
            "hashtags":
                _extract_hashtags(snippet),
            "views":
                int(stats.get("viewCount", 0)),
            "likes":
                int(stats.get("likeCount", 0)),
            "comments":
                int(stats.get("commentCount", 0)),
            "channel_account":
                snippet.get("channelTitle", ""),
            "download_path": "",
            "scraped_at": scraped_at
        })

    current_df = pd.DataFrame(
        rows
    )

    # ---------------------------------------------
    # Current weekly snapshot
    # ---------------------------------------------

    latest_path = os.path.join(
        data_dir,
        "latest_youtube_trends.csv"
    )

    current_df.to_csv(
        latest_path,
        index=False
    )

    # ---------------------------------------------
    # PROCESSED IDs
    #
    # Only dashboard_data.csv counts as completed.
    # ---------------------------------------------

    processed_ids = set()

    dashboard_path = os.path.join(
        data_dir,
        "dashboard_data.csv"
    )

    if os.path.exists(
        dashboard_path
    ):

        dashboard_df = pd.read_csv(
            dashboard_path
        )

        if "video_id" in dashboard_df.columns:

            processed_ids = set(
                dashboard_df[
                    "video_id"
                ]
                .dropna()
                .astype(str)
            )

    # ---------------------------------------------
    # New videos needing processing
    # ---------------------------------------------

    new_df = current_df[
        ~current_df["video_id"]
        .astype(str)
        .isin(processed_ids)
    ].copy()

    # ---------------------------------------------
    # Metadata history
    #
    # This stores the newest metadata observed
    # for each video, but DOES NOT determine
    # whether processing is complete.
    # ---------------------------------------------

    metadata_path = os.path.join(
        data_dir,
        "trending_metadata.csv"
    )

    if os.path.exists(
        metadata_path
    ):

        history_df = pd.read_csv(
            metadata_path
        )

        combined = pd.concat(
            [
                history_df,
                current_df
            ],
            ignore_index=True
        )

    else:

        combined = (
            current_df.copy()
        )

    combined["video_id"] = (
        combined["video_id"]
        .astype(str)
    )

    combined = (
        combined
        .drop_duplicates(
            subset=["video_id"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    combined.to_csv(
        metadata_path,
        index=False
    )

    print(
        "Current Morocco popular videos:",
        len(current_df)
    )

    print(
        "Already fully processed:",
        len(
            set(current_df["video_id"])
            & processed_ids
        )
    )

    print(
        "New videos requiring processing:",
        len(new_df)
    )

    print(
        "Metadata history:",
        len(combined)
    )

    return new_df
