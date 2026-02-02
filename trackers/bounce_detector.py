import numpy as np
import pandas as pd
import catboost as ctb
from scipy.spatial import distance
from scipy.interpolate import CubicSpline

class BounceDetector:
    def __init__(self, model_path=None):
        self.model = ctb.CatBoostRegressor()
        self.threshold = 0.45
        if model_path:
            self.model.load_model(model_path)
    
    def predict(self, ball_track_coords):
        """
        :param ball_track_coords: List of (x, y) tuples
        :return: List of frame indices where a bounce occurred
        """
        # Separate X and Y
        x_ball = [c[0] for c in ball_track_coords]
        y_ball = [c[1] for c in ball_track_coords]

        # 1. Smooth / Fill missing data
        x_ball, y_ball = self.smooth_predictions(x_ball, y_ball)
        
        # 2. Feature Engineering
        features, num_frames = self.prepare_features(x_ball, y_ball)
        
        # 3. Inference
        preds = self.model.predict(features)
        
        # 4. Post-processing
        ind_bounce = np.where(preds > self.threshold)[0]
        frames_bounce = []
        
        if len(ind_bounce) > 0:
            ind_bounce = self.postprocess(ind_bounce, preds)
            frames_bounce = [num_frames[x] for x in ind_bounce]
            
        return sorted(list(set(frames_bounce)))
    
    def prepare_features(self, x_ball, y_ball):
        labels = pd.DataFrame({'frame': range(len(x_ball)), 'x-coordinate': x_ball, 'y-coordinate': y_ball})
        
        num = 3
        eps = 1e-15
        
        # Lag features creation (Notebook logic)
        for i in range(1, num):
            labels[f'x_lag_{i}'] = labels['x-coordinate'].shift(i)
            labels[f'x_lag_inv_{i}'] = labels['x-coordinate'].shift(-i)
            labels[f'y_lag_{i}'] = labels['y-coordinate'].shift(i)
            labels[f'y_lag_inv_{i}'] = labels['y-coordinate'].shift(-i) 
            
            labels[f'x_diff_{i}'] = abs(labels[f'x_lag_{i}'] - labels['x-coordinate'])
            labels[f'y_diff_{i}'] = labels[f'y_lag_{i}'] - labels['y-coordinate']
            labels[f'x_diff_inv_{i}'] = abs(labels[f'x_lag_inv_{i}'] - labels['x-coordinate'])
            labels[f'y_diff_inv_{i}'] = labels[f'y_lag_inv_{i}'] - labels['y-coordinate']
            
            labels[f'x_div_{i}'] = abs(labels[f'x_diff_{i}']/(labels[f'x_diff_inv_{i}'] + eps))
            labels[f'y_div_{i}'] = labels[f'y_diff_{i}']/(labels[f'y_diff_inv_{i}'] + eps)

        # Drop NaNs created by shifting
        for i in range(1, num):
            labels = labels[labels[f'x_lag_{i}'].notna()]
            labels = labels[labels[f'x_lag_inv_{i}'].notna()]
        labels = labels[labels['x-coordinate'].notna()] 
        
        # Select Feature Columns
        colnames_x = [f'x_diff_{i}' for i in range(1, num)] + \
                     [f'x_diff_inv_{i}' for i in range(1, num)] + \
                     [f'x_div_{i}' for i in range(1, num)]
        colnames_y = [f'y_diff_{i}' for i in range(1, num)] + \
                     [f'y_diff_inv_{i}' for i in range(1, num)] + \
                     [f'y_div_{i}' for i in range(1, num)]
        colnames = colnames_x + colnames_y

        features = labels[colnames]
        return features, list(labels['frame'])

    def smooth_predictions(self, x_ball, y_ball):
        # Fill short gaps in tracking
        is_none = [int(x is None) for x in x_ball]
        interp = 5
        counter = 0
        for num in range(interp, len(x_ball)-1):
            if not x_ball[num] and sum(is_none[num-interp:num]) == 0 and counter < 3:
                x_ext, y_ext = self.extrapolate(x_ball[num-interp:num], y_ball[num-interp:num])
                x_ball[num] = x_ext
                y_ball[num] = y_ext
                is_none[num] = 0
                if x_ball[num+1]:
                    dist = distance.euclidean((x_ext, y_ext), (x_ball[num+1], y_ball[num+1]))
                    if dist > 80:
                        x_ball[num+1], y_ball[num+1], is_none[num+1] = None, None, 1
                counter += 1
            else:
                counter = 0  
        return x_ball, y_ball

    def extrapolate(self, x_coords, y_coords):
        xs = list(range(len(x_coords)))
        func_x = CubicSpline(xs, x_coords, bc_type='natural')
        x_ext = func_x(len(x_coords))
        func_y = CubicSpline(xs, y_coords, bc_type='natural')
        y_ext = func_y(len(x_coords))
        return float(x_ext), float(y_ext)    

    def postprocess(self, ind_bounce, preds):
        # Filter consecutive frames to find the peak probability
        ind_bounce_filtered = [ind_bounce[0]]
        for i in range(1, len(ind_bounce)):
            if (ind_bounce[i] - ind_bounce[i-1]) != 1:
                cur_ind = ind_bounce[i]
                ind_bounce_filtered.append(cur_ind)
            elif preds[ind_bounce[i]] > preds[ind_bounce[i-1]]:
                ind_bounce_filtered[-1] = ind_bounce[i]
        return ind_bounce_filtered