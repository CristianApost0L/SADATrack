#!/usr/bin/env python3
"""
Tennis Ball Tracking and Analysis Pipeline
==========================================

This script demonstrates the complete pipeline for tennis ball tracking,
bounce detection, court mapping, and match analysis.

Usage:
    python scripts/full_pipeline.py --video path/to/video.mp4 --config config.yaml
"""

import argparse
import os
import sys
import yaml
import torch
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src import (
    BallDetector, 
    CourtDetectorNet, 
    BounceDetector, 
    OptimizedBounceDetector,
    PersonDetector,
    read_video,
    PreciseCoordinateExtractor,
    create_heatmap_overlay_video,
    evaluate_models
)

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_device():
    """Setup computation device"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    return device

def main():
    parser = argparse.ArgumentParser(description="Tennis Ball Tracking Pipeline")
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--ball-model', type=str, help='Path to ball tracking model')
    parser.add_argument('--court-model', type=str, help='Path to court detection model')
    parser.add_argument('--bounce-model', type=str, help='Path to bounce detection model')
    parser.add_argument('--detector-type', type=str, choices=['catboost', 'physics'], 
                        default='physics', help='Bounce detector type')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config file {args.config} not found, using defaults")
        config = {}
    
    # Setup
    device = setup_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load video
    print("=== Step 1: Loading Video ===")
    frames, fps = read_video(args.video)
    print(f"Loaded {len(frames)} frames at {fps} FPS")
    
    # Setup models
    ball_model_path = args.ball_model or config.get('models', {}).get('ball_tracker')
    court_model_path = args.court_model or config.get('models', {}).get('court_detector')
    bounce_model_path = args.bounce_model or config.get('models', {}).get('bounce_detector')
    
    # Step 2: Ball Detection
    print("\n=== Step 2: Ball Tracking ===")
    if ball_model_path and os.path.exists(ball_model_path):
        ball_detector = BallDetector(ball_model_path, device)
        ball_track = ball_detector.infer_model(frames)
        print(f"Tracked ball across {len(ball_track)} frames")
    else:
        print("Warning: Ball model not found, using mock data")
        ball_track = [(None, None)] * len(frames)
    
    # Step 3: Court Detection  
    print("\n=== Step 3: Court Detection ===")
    if court_model_path and os.path.exists(court_model_path):
        court_detector = CourtDetectorNet(court_model_path, device)
        homography_matrices, kps_court = court_detector.infer_model(frames)
        print(f"Detected court in {sum(1 for x in homography_matrices if x is not None)} frames")
    else:
        print("Warning: Court model not found, using mock data")
        homography_matrices = [None] * len(frames)
        kps_court = [None] * len(frames)
    
    # Step 4: Bounce Detection
    print("\n=== Step 4: Bounce Detection ===")
    if args.detector_type == 'catboost' and bounce_model_path and os.path.exists(bounce_model_path):
        bounce_detector = BounceDetector(bounce_model_path)
        x_ball = [x[0] for x in ball_track]
        y_ball = [x[1] for x in ball_track]
        bounces = bounce_detector.predict(x_ball, y_ball)
        print(f"Detected {len(bounces)} bounces using CatBoost model")
    else:
        # Use physics-based detector
        bounce_detector = OptimizedBounceDetector(fps=fps, verbose=True)
        bounces_list, hits, y_smooth, vx = bounce_detector.predict(ball_track)
        bounces = set(bounces_list)
        print(f"Detected {len(bounces)} bounces using physics-based model")
    
    # Step 5: Coordinate Extraction
    print("\n=== Step 5: Coordinate Extraction ===")
    extractor = PreciseCoordinateExtractor()
    df_results = extractor.get_coordinates(bounces, ball_track, homography_matrices)
    print(f"Extracted coordinates for {len(df_results)} bounces")
    
    # Step 6: Save Results
    print("\n=== Step 6: Saving Results ===")
    
    # Save bounce coordinates
    results_file = output_dir / 'bounce_coordinates.csv'
    df_results.to_csv(results_file, index=False)
    print(f"Saved bounce coordinates to: {results_file}")
    
    # Save ball tracking data
    ball_data = []
    for i, (x, y) in enumerate(ball_track):
        ball_data.append({
            "frame": i,
            "time_sec": i / fps,
            "x_pixel": x,
            "y_pixel": y,
            "detected": x is not None
        })
    
    df_ball = pd.DataFrame(ball_data)
    ball_file = output_dir / 'ball_tracking.csv'
    df_ball.to_csv(ball_file, index=False)
    print(f"Saved ball tracking data to: {ball_file}")
    
    # Step 7: Generate Visualization Video (optional)
    if config.get('generate_video', False) and len(df_results) > 0:
        print("\n=== Step 7: Generating Visualization Video ===")
        video_output = output_dir / 'analysis_video.mp4'
        try:
            create_heatmap_overlay_video(
                args.video,
                str(video_output),
                df_results,
                homography_matrices
            )
            print(f"Saved visualization video to: {video_output}")
        except Exception as e:
            print(f"Error generating video: {e}")
    
    # Summary
    print("\n=== Pipeline Summary ===")
    print(f"Video: {args.video}")
    print(f"Frames processed: {len(frames)}")
    print(f"Bounces detected: {len(bounces)}")
    print(f"Valid coordinates extracted: {len(df_results)}")
    
    if len(df_results) > 0:
        print("\nBounce Distribution:")
        print(df_results.groupby(['side', 'depth']).size().reset_index(name='count'))
    
    print(f"\nResults saved to: {output_dir}")
    
    return df_results

if __name__ == "__main__":
    main()