%%writefile /kaggle/working/CV_project/simple_main.py
import cv2
import os
import argparse
from utils import read_video, save_video
from trackers import PlayerTracker
from court_line_detector import CourtLineDetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to input video")
    parser.add_argument("--player-detection-court-margin", type=int, default=50)
    # Ignored arguments to prevent crashes if passed
    parser.add_argument("--window-size", type=int, default=40) 
    args = parser.parse_args()

    # 1. Setup Paths
    # CORRECTED MODEL PATH HERE
    player_model_path = '/kaggle/input/cv-project/yolo26x.pt'
    court_model_path = "models/keypoints_model.pth" 

    # 2. Read Video
    video_frames = read_video(args.path)
    if not video_frames:
        print(f"❌ Error: No frames found in {args.path}")
        return

    # 3. Detect Court (First frame only)
    print("   🎾 Detecting Court Lines...")
    court_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_detector.predict(video_frames[0])

    # 4. Detect Players
    print(f"   🏃 Detecting Players using {player_model_path}...")
    player_tracker = PlayerTracker(model_path=player_model_path)
    
    # read_from_stub=False forces detection instead of looking for a saved stub file
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=False)

    # 5. Filter Players (remove ball boys/spectators)
    print("   🛡️  Filtering Players...")
    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints, 
        player_detections, 
        args.player_detection_court_margin
    )

    # 6. Draw Annotations
    print("   ✏️  Drawing results...")
    video_frames = court_detector.draw_keypoints_on_video(video_frames, court_keypoints)
    video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    # 7. Save Output
    os.makedirs("output_videos", exist_ok=True)
    filename = os.path.basename(args.path)
    output_path = os.path.join("output_videos", filename)
    save_video(video_frames, output_path)
    print(f"   ✅ Saved video to {output_path}")

if __name__ == "__main__":
    main()