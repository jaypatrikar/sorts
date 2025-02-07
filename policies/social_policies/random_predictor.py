# --------------------------------------------------------------------------------------------------
# @file:    random_predictor.py
# @brief:   A class for generating a random probability distribution.
# --------------------------------------------------------------------------------------------------
import numpy as np

from scipy.stats import norm
from scipy.special import softmax
from typing import List

from utils.common import Config
from gym.gym import Agent

class RandomPolicy:
    """ Implements a random probability distribution predictor. """
    def __init__(self, config: Config) -> None:
        self.config = config

        self.num_predictions = self.config.SOCIAL_POLICY["num_predictions"]
        super().__init__()
        
    def compute_social_action(self, agents: List[Agent], S: np.array, current_agent: int):
        """ Obtains actions probablities using random sampling.
        
        Inputs
        ------
        num_episode[int]: id of episode that will run. 
        """
        action_size = S.shape[0]

        action_probs = np.zeros(shape=(action_size, 1))
        for i in range(1):
            rvs = norm.rvs(size=action_size)
            ap = rvs - np.min(rvs)
            ap /= np.sum(ap)
            action_probs[:, i] = ap
        return np.squeeze(action_probs)
    
    def compute_goal_action(self, agents: List[Agent], S: np.array, current_agent: int):
        g = [[-200,0],[200,0]]

        val = np.linalg.norm(S[-1] - g[current_agent])


        return softmax(-val) 



