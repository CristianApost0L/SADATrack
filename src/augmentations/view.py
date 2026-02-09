"""
View-based augmentations (3D rotations and camera simulations).
"""
import numpy as np


# augment_3d_view_rotation() has been removed
# This function required 3D coordinates (Z) from MotionBERT which is no longer used.
# With only 2D YOLO keypoints (X, Y, Conf), 3D rotations don't make sense.


# simulate_back_view() has been removed
# YOLO-Pose already detects people from back view naturally,
# so synthetic back view simulation is redundant and can introduce confusion.
