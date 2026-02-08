import os
import json
import constants
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def export_json(player_id_map, player_detections, original_video_name): 
    '''
    Exports player detection data and ID mappings to a JSON file for debugging and analysis.

    This function serializes the detection data, handling NumPy types automatically.
    It saves the output to the directory specified in `constants.JSON_OUTPUT_DIR`.

    Args:
        player_id_map (dict): A mapping of Tracking IDs to Logical Player IDs 
                              (e.g., {5: 1, 8: 2}).
        player_detections (list): A list of dictionaries (one per frame) containing 
                                  bounding boxes for each tracked ID.
        original_video_name (str or tuple): The name of the original video file. 
                                            Can be a string (filename) or a tuple returned 
                                            by `os.path.splitext` ((root, ext)).

    Returns:
        None: The function writes to disk and prints a status message.
    '''
    try:
        # 1. Create a structured dictionary including the Map and Frames
        export_data = {
            "player_id_map": player_id_map,  # <--- The new requirement
            "frames": {}
        }
        
        # 2. Fill in the frame data
        for i, frame_detections in enumerate(player_detections):
            frame_boxes = []
            for track_id, data in frame_detections.items():
                keypoints = data.get('keypoints', [])
                frame_boxes.append({
                    "track_id": track_id,
                    "bbox": data['bbox'],
                    "keypoints": keypoints
                })
            # Use string keys for frames to be valid JSON
            export_data["frames"][i] = frame_boxes

        # 3. Setup output folder
        json_output_dir = constants.JSON_OUTPUT_DIR
        os.makedirs(json_output_dir, exist_ok=True)
        
        # 4. Handle filename safely (Tuple check logic)
        if isinstance(original_video_name, tuple):
            file_root = original_video_name[0]
        else:
            file_root = os.path.splitext(os.path.basename(str(original_video_name)))[0]

        json_output_path = os.path.join(json_output_dir, f"{file_root}_detections.json")

        with open(json_output_path, 'w') as f:
            json.dump(export_data, f, cls=NumpyEncoder, indent=4)
        print(f"   📄 Saved detection log + ID Map to {json_output_path}")

    except Exception as e:
        print(f"   ⚠️ Could not save detection JSON: {e}")