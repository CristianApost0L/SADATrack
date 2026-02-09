# Tennis Ball Tracking and Analysis

A comprehensive computer vision system for tennis ball tracking, bounce detection, court mapping, and match analysis implemented in a Jupyter notebook environment.

## Output & Results

The notebook generates various outputs for analysis:

1. **Ball Tracking Results**:
   - Frame-by-frame ball positions (pixel coordinates)
   - Detection confidence scores
   - Interpolated positions for missing detections

2. **Court Analysis**:
   - Court line keypoints and homography matrices
   - Real-world coordinate mappings  
   - Court zone classifications

3. **Bounce Detection**:
   - Bounce timestamps and locations
   - Trajectory analysis and smoothing results
   - Physics-based validation metrics

4. **Visualizations**:
   - Video overlays with tracking results
   - Interactive plots and heatmaps
   - Statistical analysis charts


## Dataset and Checkpoints

Datasets used:
- [Tennis Bounces Ground Truth](https://www.kaggle.com/datasets/soykataminsapienza/tennis-bounces-gt)
- [Tennis Bounce Dataset](https://www.kaggle.com/datasets/soykataminsapienza/tennis-bounce-dataset) 
- [Tennis Rallies](https://www.kaggle.com/datasets/soykataminsapienza/tennis-rallies/data)

[Kosolapov Sergey's](https://github.com/yastrebksv/TennisProject) checkpoints:
- [Tennis Models Checkpoints](https://www.kaggle.com/models/soykataminsapienza/tennismodels/pyTorch/default/1)

## Acknowledgments

- TrackNet architecture based on the original TrackNet paper for sports ball tracking
- Court detection algorithm and initial bounce detection based on [Kosolapov Sergey's project](https://medium.com/@kosolapov.aetp/tennis-analysis-using-deep-learning-and-machine-learning-a5a74db7e2ee).
