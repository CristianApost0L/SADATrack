SINGLE_LINE_WIDTH = 8.23
HALF_COURT_LINE_HEIGHT = 11.88
SERVICE_LINE_WIDTH = 6.4
DOUBLE_ALLY_DIFFERENCE = 1.37
NO_MANS_LAND_HEIGHT = 5.48

DOUBLE_LINE_WIDTH = 10.97 # The only fixed measurement we actually need

BOUNDING_BOX_PADDING = 30 # Pixels of padding around the player bounding box

# The official 12 shot type classes from the dataset
THETIS_CLASSES = [
    "backhand2hands", "backhand", "backhand_slice", "backhand_volley",
    "forehand_flat", "forehand_openstands", "forehand_slice", "forehand_volley",
    "flat_service", "kick_service", "slice_service", "smash"
]

TARGET_FPS = 24 # Target FPS for videos

COURT_INFER_INTERVAL = 12 # Refresh court keypoints every half second (24 FPS videos)