import catboost as ctb
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import distance

class BounceDetector:
    def __init__(self, path_model=None):
        self.model = ctb.CatBoostRegressor()
        self.threshold = 0.45
        if path_model:
            self.load_model(path_model)
        
    def load_model(self, path_model):
        self.model.load_model(path_model)
    
    def prepare_features(self, x_ball, y_ball):
        labels = pd.DataFrame({'frame': range(len(x_ball)), 'x-coordinate': x_ball, 'y-coordinate': y_ball})
        
        num = 3
        eps = 1e-15
        for i in range(1, num):
            labels['x_lag_{}'.format(i)] = labels['x-coordinate'].shift(i)
            labels['x_lag_inv_{}'.format(i)] = labels['x-coordinate'].shift(-i)
            labels['y_lag_{}'.format(i)] = labels['y-coordinate'].shift(i)
            labels['y_lag_inv_{}'.format(i)] = labels['y-coordinate'].shift(-i) 
            labels['x_diff_{}'.format(i)] = abs(labels['x_lag_{}'.format(i)] - labels['x-coordinate'])
            labels['y_diff_{}'.format(i)] = labels['y_lag_{}'.format(i)] - labels['y-coordinate']
            labels['x_diff_inv_{}'.format(i)] = abs(labels['x_lag_inv_{}'.format(i)] - labels['x-coordinate'])
            labels['y_diff_inv_{}'.format(i)] = labels['y_lag_inv_{}'.format(i)] - labels['y-coordinate']
            labels['x_div_{}'.format(i)] = abs(labels['x_diff_{}'.format(i)]/(labels['x_diff_inv_{}'.format(i)] + eps))
            labels['y_div_{}'.format(i)] = labels['y_diff_{}'.format(i)]/(labels['y_diff_inv_{}'.format(i)] + eps)

        for i in range(1, num):
            labels = labels[labels['x_lag_{}'.format(i)].notna()]
            labels = labels[labels['x_lag_inv_{}'.format(i)].notna()]
        labels = labels[labels['x-coordinate'].notna()] 
        
        colnames_x = ['x_diff_{}'.format(i) for i in range(1, num)] + \
                     ['x_diff_inv_{}'.format(i) for i in range(1, num)] + \
                     ['x_div_{}'.format(i) for i in range(1, num)]
        colnames_y = ['y_diff_{}'.format(i) for i in range(1, num)] + \
                     ['y_diff_inv_{}'.format(i) for i in range(1, num)] + \
                     ['y_div_{}'.format(i) for i in range(1, num)]
        colnames = colnames_x + colnames_y

        features = labels[colnames]
        return features, list(labels['frame'])
    
    def predict(self, x_ball, y_ball, smooth=True):
        if smooth:
            x_ball, y_ball = self.smooth_predictions(x_ball, y_ball)
        features, num_frames = self.prepare_features(x_ball, y_ball)
        preds = self.model.predict(features)
        ind_bounce = np.where(preds > self.threshold)[0]
        if len(ind_bounce) > 0:
            ind_bounce = self.postprocess(ind_bounce, preds)
        frames_bounce = [num_frames[x] for x in ind_bounce]
        return set(frames_bounce)
    
    def smooth_predictions(self, x_ball, y_ball):
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
        ind_bounce_filtered = [ind_bounce[0]]
        for i in range(1, len(ind_bounce)):
            if (ind_bounce[i] - ind_bounce[i-1]) != 1:
                cur_ind = ind_bounce[i]
                ind_bounce_filtered.append(cur_ind)
            elif preds[ind_bounce[i]] > preds[ind_bounce[i-1]]:
                ind_bounce_filtered[-1] = ind_bounce[i]
        return ind_bounce_filtered


class OptimizedBounceDetector:
    def __init__(self, fps=30, y_thresh=75, verbose=True):
        self.fps = fps
        self.smooth_window = 3       
        self.prominence = 1.0        
        self.offset_correction = -4 
        self.y_thresh = y_thresh     
        self.verbose = verbose       
        
    def predict(self, ball_track):
        from scipy.signal import savgol_filter, find_peaks
        
        # --- 1. PREPROCESSING ---
        y_raw = [p[1] if p is not None else np.nan for p in ball_track]
        x_raw = [p[0] if p is not None else np.nan for p in ball_track]
        
        y_filled = pd.Series(y_raw).interpolate(method='linear', limit_direction='both').to_numpy()
        x_filled = pd.Series(x_raw).interpolate(method='linear', limit_direction='both').to_numpy()
        
        # SENSITIVE SMOOTHING
        y_smooth = savgol_filter(y_filled, window_length=self.smooth_window, polyorder=2)
        x_smooth = savgol_filter(x_filled, window_length=self.smooth_window, polyorder=2)

        vx = np.gradient(x_smooth) 
        
        # --- 2. CANDIDATE GENERATION ---
        # Find every local maximum in Y (lowest point on screen)
        candidates, _ = find_peaks(y_smooth, distance=5, prominence=self.prominence)
        
        if self.verbose:
            print(f"DEBUG: Found {len(candidates)} raw candidates.")
        
        confirmed_bounces = []
        rejected_hits = [] 
        
        for idx in candidates:
            if idx < 3 or idx >= len(y_smooth) - 3: continue
            
            # --- FILTER 1: COURT POSITION (FIXED) ---
            # Rejection: Only ignore if it's REALLY high (Sky/Crowd)
            if y_smooth[idx] < self.y_thresh: 
                if self.verbose and len(confirmed_bounces) < 3: 
                    print(f"  - Cand {idx} rejected: Too High (Y={y_smooth[idx]:.1f} < {self.y_thresh})")
                continue 

            # --- FILTER 2: HORIZONTAL DIRECTION ---
            vx_in = np.mean(vx[idx-3:idx])
            vx_out = np.mean(vx[idx:idx+3])
            is_moving_horizontally = (abs(vx_in) > 0.5) or (abs(vx_out) > 0.5)
            
            if is_moving_horizontally:
                # Reversal = Hit
                if (vx_in * vx_out) < -0.5:
                    rejected_hits.append(idx)
                    continue 

            # --- FILTER 3: VERTICAL PHYSICS SANITY ---
            vy = np.gradient(y_smooth)
            vy_in = np.mean(vy[idx-3:idx])
            
            # Moving UP before the event? Not a bounce.
            if vy_in < -2: 
                continue

            # --- SUCCESS ---
            corrected_frame = idx + self.offset_correction
            confirmed_bounces.append(int(corrected_frame))
            
        if self.verbose:
            print(f"DEBUG: Final confirmed bounces: {len(confirmed_bounces)}")
            
        return sorted(list(set(confirmed_bounces))), rejected_hits, y_smooth, vx