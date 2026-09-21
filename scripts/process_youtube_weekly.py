
import os
import subprocess
import pandas as pd
import whisper

from utils.config import (
    DATA_DIR,
    YOUTUBE_AUDIO_DIR,
    YOUTUBE_TRANSCRIPTS_DIR
)


def process_new_youtube_videos(new_df):
    """
    Automatically process only NEW YouTube videos.

    For each new video:
    1. Skip if transcript already exists.
    2. Download audio temporarily.
    3. Transcribe with Whisper.
    4. Save transcript as .txt.
    5. Delete temporary audio.
    """

    if new_df is None or len(new_df) == 0:
        print("No new videos to process ✅")
        return new_df

    os.makedirs(YOUTUBE_AUDIO_DIR, exist_ok=True)
    os.makedirs(YOUTUBE_TRANSCRIPTS_DIR, exist_ok=True)

    new_df = new_df.copy()

    if "transcript" not in new_df.columns:
        new_df["transcript"] = ""

    # Load Whisper only after an audio download succeeds
    model = None
    
    success = 0
    skipped = 0
    failed = 0

    for index, row in new_df.iterrows():

        video_id = str(row["video_id"])
        url = row["url"]

        transcript_path = os.path.join(
            YOUTUBE_TRANSCRIPTS_DIR,
            f"{video_id}.txt"
        )

        print("\n" + "=" * 60)
        print(video_id)
        print(row["title_caption"][:100])

        # --------------------------------------------
        # Already processed
        # --------------------------------------------

        if os.path.exists(transcript_path):

            with open(
                transcript_path,
                "r",
                encoding="utf-8"
            ) as f:
                transcript = f.read().strip()

            new_df.at[index, "transcript"] = transcript

            skipped += 1
            print("Transcript already exists ✅")
            continue

        # --------------------------------------------
        # Temporary audio path
        # --------------------------------------------

        output_template = os.path.join(
            YOUTUBE_AUDIO_DIR,
            f"{video_id}.%(ext)s"
        )

        command = [
            "yt-dlp",
            "-x",
            "--audio-format", "mp3",
            "--audio-quality", "5",
            "-o", output_template,
            url
        ]

        try:

            print("Downloading audio...")

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:

                failed += 1
                print("Audio download failed ❌")
                print(result.stderr[-500:])
            
            if "Sign in to confirm you’re not a bot" in result.stderr or \
                "Sign in to confirm you're not a bot" in result.stderr:
                 print("YouTube blocked audio downloads. Switching remaining videos to metadata-only.")
                 break

             continue

            # Find downloaded audio
            audio_files = [
                f for f in os.listdir(YOUTUBE_AUDIO_DIR)
                if f.startswith(video_id + ".")
            ]

            if not audio_files:

                failed += 1
                print("Audio file not found ❌")
                continue

            audio_path = os.path.join(
                YOUTUBE_AUDIO_DIR,
                audio_files[0]
            )

             # --------------------------------------------
            # Whisper
            # --------------------------------------------

            if model is None:
                print("Loading Whisper model...")
                model = whisper.load_model("base")
                print("Whisper ready ✅")

            print("Transcribing...")

            transcription = model.transcribe(
                audio_path
            )

            transcript = (
                transcription.get("text", "")
                .strip()
            )

            # Save transcript even if empty so we know
            # the video was attempted.
            with open(
                transcript_path,
                "w",
                encoding="utf-8"
            ) as f:
                f.write(transcript)

            new_df.at[index, "transcript"] = transcript

            success += 1

            print(
                f"Transcript saved ✅ "
                f"({len(transcript)} characters)"
            )

        except Exception as e:

            failed += 1
            print("Processing error ❌")
            print(e)

        finally:

            # --------------------------------------------
            # Remove temporary media
            # --------------------------------------------

            for file in os.listdir(YOUTUBE_AUDIO_DIR):

                if file.startswith(video_id + "."):

                    try:
                        os.remove(
                            os.path.join(
                                YOUTUBE_AUDIO_DIR,
                                file
                            )
                        )
                    except Exception:
                        pass

    print("\n" + "=" * 60)
    print("WEEKLY TRANSCRIPTION SUMMARY")
    print("=" * 60)
    print("New transcripts:", success)
    print("Already processed:", skipped)
    print("Failed:", failed)

    return new_df
