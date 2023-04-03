# --------------------------------------------------------------------------------------------------
# @file:    multi_agent.py
# @brief:   Class that inherits from SelfPlay, and implements the procedure for runnning multi-agent
#           episodes for self-play simulation. 
# --------------------------------------------------------------------------------------------------
import numpy as np
import os

from sorts.utils.common import Config
from gym.gym import Gym
import logging
import torch
class IsaacHIL:
    """ Runs self-play experiments in multi-agent setting. """
    def __init__(self, config: Config) -> None:
        """ Intializes game play. 
        
        Inputs
        ------
            config[dict]: dictionary with configuration parameters. 
        """
        # super().__init__(config=config)     
        self._config =  Config(config)
        self.setup()

        self.logger.info(f"{self.name} is ready!")

    @property
    def config(self) -> Config:
        return self._config
    @property
    def name(self) -> str:
        return self.__class__.__name__
       
    def setup(self) -> None:
        """ Setup process which includes: 
                * Initialize logger;
                * Create the experiment name-tag;
                * Create output directories, and;
                * Initialize gym, search policy and relevant member parameters. 
        """
        # create the experiment tag name
        exp_name = "policy-{}-{}_num-agents-{}_num-episodes-{}".format(
            self.config.PLANNER_POLICY.type, 
            self.config.SOCIAL_POLICY.type, 
            self.config.GAME.num_agents, 
            self.config.GAME.num_episodes,
        )
        # if self.config.PLANNER_POLICY.type == "mcts":
        #     exp_name = "PLANNER-{}-{}-{}-{}_SOCIAL-{}-{}_GAME-{}-{}-{}".format(
        #         self.config.PLANNER_POLICY.type, 
        #         self.config.PLANNER_POLICY.c_uct,
        #         self.config.PLANNER_POLICY.h_uct,
        #         self.config.PLANNER_POLICY.num_ts,

        #         self.config.SOCIAL_POLICY.type, 
        #         self.config.SOCIAL_POLICY.num_predictions,

        #         self.config.GAME.num_agents, 
        #         self.config.GAME.num_episodes,
        #         self.config.GAME.max_steps,
        #     )
        
        # output directory and logging file

        self.logger = logging.getLogger(__name__)
        self.out = os.path.join(self.config.MAIN.out_dir, self.config.DATA.dataset_name, exp_name)

        # create environment and planner policy
        self.logger.info(f"Initializing environment and search policy.")
        self.gym = Gym(self.config, self.logger, self.out)
        
        # supported planners
        if self.config.PLANNER_POLICY.type == "mcts":
            from policies.planner_policies.mcts import MCTS as Planner 
        elif self.config.PLANNER_POLICY.type == "baseline":
            from policies.planner_policies.baseline import Baseline as Planner
        else:
            raise NotImplementedError(f"Policy {self.config.PLANNER_POLICY.type} not supported!")
        
        self.policy = Planner(self.config, self.gym, self.logger)
    
        # initialize base parameters
        self.num_episodes = self.config.GAME.num_episodes
        self.num_agents = self.config.GAME.num_agents
        self.gym.playing = [0, 1] if self.num_agents > 1 else [0]
        
        self.agents = self.gym.spawn_agents(scheme="hil")    
        assert len(self.agents) > 1, f"Invalid number of agents: {len(self.agents)}"
        self.gym.reset()
        self.policy.reset()

    def run_step(self,curr_state):
        action = 0
        if self.config.VISUALIZATION.visualize:
            self.gym.show_world(self.agents,show_tree=True)


        self.agents[0].step(curr_state[0])
        self.agents[1].step(curr_state[1])
        pi = self.policy.compute_action_probabilities(agents=self.agents)

        # if self.config.GAME.deterministic:
        #     action = np.argmax(pi)
        # else: # sample
        #     action = np.random.choice(self.gym.action_size, size=1, p=pi)[0]

        # new_state = self.gym.get_next_state(self.agents[current_agent].trajectory[-1], action)

        return action

