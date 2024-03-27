from hw4.roble.infrastructure.gclr_wrapper import GoalConditionedEnv, GoalConditionedEnvV2

import torch

class HRLWrapper(GoalConditionedEnv):

    def __init__(self, env, params, policy):
        # TODO
        ## Load the policy \pi(a|s,g,\theta^{low}) you trained from Q4.
        ## Make sure the policy you load was trained with the same goal_frequency
        super(GoalConditionedEnv, self).__init__(env, params)  
        self.low_policy = torch.load(policy)

    def step(self, action):
        ## Add code to compute a new goal-conditioned reward
        # TODO
        sub_goal = action # The high level policy action \pi(g|s,\theta^{hi}) is the low level goal.
        for i in range(self.params['env']['goal_frequency']):
            ## Get the action to apply in the environment
            ## HINT you need to use \pi(a|s,g,\theta^{low})
            ## Step the environment
            action = self.low_policy(self.createState(self.obs, sub_goal))
            self.obs, reward, self.done, self.info = self.env.step(action)
            
        # return s_{t+k}, r_{t+k}, done, info
        self.info["reached_goal"] = self.success_fn()
        return self.obs, self.reward(self.obs, action), self.done, self.info