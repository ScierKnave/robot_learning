import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import gym
from hw1.roble.infrastructure import pytorch_util as ptu


class GoalConditionedEnv(object):

    def __init__(self, env, params):
        # snatch attributes from original env
        self.env = env
        self.params = params   
        self.unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env

        if self.params['env']['env_name'] == 'reacher':
            self.additional_dims = 3
            self.goal_indicies =[-6, -5, -4]
            self.uniform_bounds = torch.Tensor([[-0.6, -1.4, -0.4], [0.8, 0.2, 0.5]])
            self.gaussian_bounds = torch.Tensor([[ 0.2,-0.7,0.0], [0.3, 0.4, 0.05]])
            self.bound_var = 3
            self.k = 15
            self.goal_reached_threshold = -0.3

        if self.params['env']['env_name'] == 'antmaze':
            self.additional_dims = 2
            self.goal_indicies = [0,1]
            self.uniform_bounds = torch.Tensor([[-4,-4],[20,4]])
            self.gaussian_bounds = torch.Tensor([[0,8],[4,4]])
            self.bound_var = 0.3
            self.k = 15
            self.goal_reached_threshold = -5.0

        if self.params['env']['env_name'] == 'widowx':
            self.additional_dims = 3
            self.goal_indicies = [0,1,2]
            self.uniform_bounds = torch.Tensor([[0.4,-0.2,-0.34],[0.8,0.4,-0.1]])
            self.gaussian_bounds = torch.Tensor([[0.6,0.1,-0.2], [0.2,0.2,0.2]])
            self.bound_var = 0.3
            self.k = 15
            self.goal_reached_threshold = -0.3

        self.reacher_nb_dims = 3
        self.ant_nb_dims = 2
        self.widowx_nb_dims = 3
        

    def render(self, mode='human', **kwargs):
        # pass render from sub env
        return self.env.render(mode=mode, **kwargs)
    
    def seed(self, seed):
        #TODO
        return seed
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        # Create a new observation space with the modified dimensions
        modified_observation_space = gym.spaces.Box(
            low=np.append(self.env.observation_space.low, -np.inf * np.ones(self.additional_dims)),
            high=np.append(self.env.observation_space.high, np.inf * np.ones(self.additional_dims)),
            dtype=self.env.observation_space.dtype
        )
        return modified_observation_space

    @property
    def metadata(self):
        return self.env.metadata
    
    def sampleGoal(self):

        if self.params['env']['goal_rep'] == 'absolute':

            if self.params['env']['goal_dist'] == 'uniform':
                low = self.uniform_bounds[0]
                high = self.uniform_bounds[1]
                self.goal_dist = torch.distributions.uniform.Uniform(low, high)
            
            if self.params['env']['goal_dist'] == 'normal':
                mean = self.gaussian_bounds[0]
                std = self.gaussian_bounds[1]
                self.goal_dist = torch.distributions.normal.Normal(mean, std)


        if self.params['env']['goal_rep'] == 'relative':

            if self.params['env']['goal_dist'] == 'uniform':
                #low = torch.Tensor(self.params['env']['goal_dist_params'][0])
                #high = torch.Tensor(self.params['env']['goal_dist_params'][1])
                #self.goal_dist = torch.distributions.uniform.Uniform(low, high)
                print('cant pick uniform here')
                exit()

            if self.params['env']['goal_dist'] == 'normal':
                pos_dim = self.getPos().shape[-1]
                var = torch.Tensor([self.bound_var * pos_dim])
                self.goal_dist = torch.distributions.normal.Normal(torch.Tensor(self.getPos()), var)

        self.goal = self.goal_dist.sample()


        if self.params['env']['env_name'] == 'reacher': # TODO: verify
            self.env.model.site_pos[self.env.target_sid] = self.goal.numpy()
            self.env.sim.forward()

        return self.goal.numpy()

    def success_fn(self):
        return self.reward(self.obs) < self.goal_reached_threshold


    def getPos(self):
        return self.obs[..., self.goal_indicies]

    def getGoalDescription(self):
        if self.params['env']['goal_rep'] == 'absolute': return self.goal
        return self.getPos() - self.goal
    
    def reset(self):
        # Add code to generate a goal from a distribution
        self.obs = self.env.reset()
        self.goal = self.sampleGoal()
        return self.createState(self.obs)

    def reward(self, obs, action=None):
        # reward is euclidean norm between goal pos and current pos
        return -np.linalg.norm(obs[..., self.goal_indicies] - self.goal, axis=-1)


    def step(self, action):
        ## Add code to compute a new goal-conditioned reward
        self.obs, reward, self.done, self.info = self.env.step(action)
        self.info["reached_goal"] = self.success_fn()
        return self.createState(self.obs), self.reward(self.obs, action), self.done, self.info
        
    def createState(self, obs):
        ## Add the goal to the state
        return np.concatenate([obs, self.getGoalDescription()])
        
class GoalConditionedEnvV2(GoalConditionedEnv):

    def __init__(self, env, params):
        super().__init__(env, params)
        self.step_count = 0
    
    def reset(self):
        # Add code to generate a goal from a distribution
        self.step_count = 0
        return super().reset()

    def step(self, action):
        ## Add code to compute a new goal-conditioned reward
        obs, reward, done, info = super().step(action)
        self.step_count += 1
        if self.step_count % self.params['env']['goal_frequency'] == 0:
            self.sampleGoal()
        return obs, reward, done, info