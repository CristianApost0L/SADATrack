import cv2
import os
import argparse
import sys
import traceback
from utils import read_video, save_video
from trackers import PlayerTracker
from court_line_detector import CourtLineDetector

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--path", type=str, required=True, help="Path to input video")
        parser.add_argument("--player-detection-court-margin", type=int, default=50)
        parser.add_argument("--window-size", type=int, default=40) 
        args = parser.parse_args()

        # --- PATHS ---
        # Double check these exist in your kaggle input
        player_model_path = '/kaggle/input/cv-project/yolo26x.pt'
        court_model_path = "models/keypoints_model.pth" 

        print(f"DEBUG: Starting processing for {args.path}")
        print(f"DEBUG: Using Player Model: {player_model_path}")

        # 1. Read Video
        video_frames = read_video(args.path)
        if not video_frames:
            print(f"❌ Error: read_video returned 0 frames for {args.path}")
            sys.exit(1)

        # 2. Detect Court
        print("   🎾 Detecting Court Lines...")
        court_detector = CourtLineDetector(court_model_path)
        court_keypoints = court_detector.predict(video_frames[0])

        # 3. Detect Players
        print(f"   🏃 Detecting Players...")
        player_tracker = PlayerTracker(model_path=player_model_path)
        player_detections = player_tracker.detect_frames(video_frames, read_from_stub=False)

        # 4. Filter Players
        print("   🛡️  Filtering Players...")
        player_detections = player_tracker.choose_and_filter_players(
            court_keypoints, 
            player_detections, 
            args.player_detection_court_margin
        )

        # 5. Draw Annotations
        print("   ✏️  Drawing results...")
        video_frames = court_detector.draw_keypoints_on_video(video_frames, court_keypoints)
        video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

        # 6. Save Output
        os.makedirs("output_videos", exist_ok=True)
        filename = os.path.basename(args.path)
        output_path = os.path.join("output_videos", filename)
        save_video(video_frames, output_path)
        print(f"   ✅ Saved video to {output_path}")

    except Exception:
        # This ensures the actual python error is printed to the log
        print("\n\n!!!!!!!!!! PYTHON SCRIPT CRASHED !!!!!!!!!!")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()