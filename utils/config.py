
import os

# -------------------------------------------------
# Project directory detection
# -------------------------------------------------

# If running in Colab with Google Drive mounted,
# use the existing internship project folder.
COLAB_PROJECT_DIR = "/content/drive/MyDrive/morocco_trends"

if os.path.exists(COLAB_PROJECT_DIR):
    PROJECT_DIR = COLAB_PROJECT_DIR
else:
    # GitHub Actions / local deployment:
    # utils/config.py is inside PROJECT_DIR/utils/
    PROJECT_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

# -------------------------------------------------
# Project folders
# -------------------------------------------------

RAW_DATA_DIR = os.path.join(PROJECT_DIR, "raw_data")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DOWNLOADS_DIR = os.path.join(PROJECT_DIR, "downloads")
TRANSCRIPTS_DIR = os.path.join(PROJECT_DIR, "transcripts")
AUDIO_DIR = os.path.join(PROJECT_DIR, "audio")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "processed")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
HISTORY_DIR = os.path.join(PROJECT_DIR, "history")

# -------------------------------------------------
# Create folders if needed
# -------------------------------------------------

for folder in [
    RAW_DATA_DIR,
    DATA_DIR,
    DOWNLOADS_DIR,
    TRANSCRIPTS_DIR,
    AUDIO_DIR,
    PROCESSED_DIR,
    LOGS_DIR,
    HISTORY_DIR
]:
    os.makedirs(folder, exist_ok=True)


# -------------------------------------------------
# YouTube weekly automation folders
# -------------------------------------------------

YOUTUBE_DOWNLOADS_DIR = os.path.join(DOWNLOADS_DIR, "youtube_weekly")
YOUTUBE_AUDIO_DIR = os.path.join(AUDIO_DIR, "youtube_weekly")
YOUTUBE_TRANSCRIPTS_DIR = os.path.join(TRANSCRIPTS_DIR, "youtube_weekly")

for folder in [
    YOUTUBE_DOWNLOADS_DIR,
    YOUTUBE_AUDIO_DIR,
    YOUTUBE_TRANSCRIPTS_DIR
]:
    os.makedirs(folder, exist_ok=True)
