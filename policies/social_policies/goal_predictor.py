# --------------------------------------------------------------------------------------------------
# @file:    random_predictor.py
# @brief:   A class for generating a random probability distribution.
# --------------------------------------------------------------------------------------------------
import numpy as np

from scipy.stats import norm
from scipy.special import softmax
from typing import List
import torch
from utils.common import Config
from gym.gym import Agent

class GoalPolicy:
    """ Implements a random probability distribution predictor. """
    def __init__(self, config: Config) -> None:
        self.config = config

        self.num_predictions = self.config.SOCIAL_POLICY["num_predictions"]
        super().__init__()
        

    
    def compute_social_action(self, agents: List[Agent], S: np.array, current_agent: int):
        g = [[-200,0],[200,0]]

        val = np.linalg.norm(S[:,-1,:2]-torch.FloatTensor(g[current_agent]),axis=1)
        val = val/np.sum(val)

        return softmax(1/val) 



