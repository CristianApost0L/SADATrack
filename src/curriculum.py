"""
Curriculum learning scheduler for progressive augmentation.
"""

class CurriculumLearningScheduler:
    """
    Gradually increase augmentation strength during training.
    Follows a curriculum: weak -> medium -> strong
    """
    def __init__(self, total_epochs, schedule_type='linear'):
        """
        Args:
            total_epochs: total number of training epochs
            schedule_type: 'linear', 'exponential', 'step'
        """
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type
        self.current_strength = 'weak'
        self.previous_strength = None
        self._strength_changed = False
        
        # Define augmentation parameters at each stage
        self.stages = {
            'weak': {
                'flip_prob': 0.3,
                'rotation_range': 5,
                'scale_range': 0.05,
                'noise_std': 0.003,
            },
            'medium': {
                'flip_prob': 0.5,
                'rotation_range': 15,
                'scale_range': 0.1,
                'noise_std': 0.005,
            },
            'strong': {
                'flip_prob': 0.7,
                'rotation_range': 25,
                'scale_range': 0.15,
                'noise_std': 0.008, 
            }
        }
    
    @staticmethod
    def get_stage_params(strength='weak'):
        """
        Get augmentation parameters for a specific strength level.
        Can be used without instantiating the scheduler.
        
        Args:
            strength: 'weak', 'medium', or 'strong'
        
        Returns:
            dict with augmentation parameters
        """
        stages = {
            'weak': {
                'flip_prob': 0.3,
                'rotation_range': 5,
                'scale_range': 0.05,
                'noise_std': 0.003,
            },
            'medium': {
                'flip_prob': 0.5,
                'rotation_range': 15,
                'scale_range': 0.1,
                'noise_std': 0.005,
            },
            'strong': {
                'flip_prob': 0.7,
                'rotation_range': 25,
                'scale_range': 0.15,
                'noise_std': 0.008,
            }
        }
        return stages.get(strength, stages['weak']).copy()
    
    def get_augmentation_params(self, current_epoch):
        """
        Get augmentation parameters for current epoch.
        
        Args:
            current_epoch: current epoch number (0-indexed)
        
        Returns:
            dict with augmentation parameters
        """
        progress = current_epoch / self.total_epochs
        
        if self.schedule_type == 'linear':
            return self._linear_interpolate(progress)
        elif self.schedule_type == 'exponential':
            return self._exponential_interpolate(progress)
        elif self.schedule_type == 'step':
            return self._step_schedule(progress)
        else:
            return self.stages['weak']
    
    def _linear_interpolate(self, progress):
        """Linear interpolation between weak and strong."""
        if progress < 0.5:
            # Weak to Medium (0 to 0.5)
            alpha = progress * 2  # 0 to 1
            return self._interpolate_dicts(self.stages['weak'], self.stages['medium'], alpha)
        else:
            # Medium to Strong (0.5 to 1)
            alpha = (progress - 0.5) * 2  # 0 to 1
            return self._interpolate_dicts(self.stages['medium'], self.stages['strong'], alpha)
    
    def _exponential_interpolate(self, progress):
        """Exponential interpolation - slower start, faster end."""
        # Exponential ease-in: faster progression toward the end
        progress_exp = progress ** 0.5  # Square root for smoother start
        
        if progress_exp < 0.5:
            alpha = progress_exp * 2
            return self._interpolate_dicts(self.stages['weak'], self.stages['medium'], alpha)
        else:
            alpha = (progress_exp - 0.5) * 2
            return self._interpolate_dicts(self.stages['medium'], self.stages['strong'], alpha)
    
    def _step_schedule(self, progress):
        """Step schedule - abrupt transitions."""
        if progress < 0.33:
            return self.stages['weak'].copy()
        elif progress < 0.66:
            return self.stages['medium'].copy()
        else:
            return self.stages['strong'].copy()
    
    def _interpolate_dicts(self, dict1, dict2, alpha):
        """Linear interpolation between two parameter dicts."""
        result = {}
        for key in dict1:
            result[key] = dict1[key] + (dict2[key] - dict1[key]) * alpha
        return result
    
    def get_current_strength(self):
        """Get current augmentation strength level as string."""
        return self.current_strength
    
    def just_changed(self):
        """Check if strength level just changed in the last step() call."""
        return self._strength_changed
    
    def step(self, current_epoch):
        """
        Update curriculum state for current epoch.
        Call this at the beginning of each epoch to track strength changes.
        
        Args:
            current_epoch: current epoch number (0-indexed)
        """
        progress = current_epoch / self.total_epochs
        
        # Determine current strength based on progress
        if self.schedule_type == 'step':
            if progress < 0.33:
                new_strength = 'weak'
            elif progress < 0.66:
                new_strength = 'medium'
            else:
                new_strength = 'strong'
        else:  # linear or exponential
            if progress < 0.5:
                new_strength = 'weak'
            elif progress < 0.75:
                new_strength = 'medium'
            else:
                new_strength = 'strong'
        
        # Detect if strength changed
        self._strength_changed = (new_strength != self.current_strength)
        
        if self._strength_changed:
            self.previous_strength = self.current_strength
            self.current_strength = new_strength

