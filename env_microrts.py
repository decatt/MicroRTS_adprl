import gym_microrts
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
import numpy as np
from utils import MaskedCategorical
import torch

class EnvMicroRTSW():
    def __init__(self, num_env:int, max_steps:int, ais:list, map_path:str, reward_weights:np.ndarray) -> None:
        self.envs = MicroRTSVecEnv(num_envs=num_env, max_steps=max_steps, ai2s=ais, map_path=map_path, reward_weight=reward_weights)
        self.num_env = num_env
        self.action_space = self.envs.action_space.nvec.tolist()

    def reset(self):
        return self.envs.reset()
    
    def step(self, actions:np.ndarray):
        return self.envs.step(actions)
    
    def render(self):
        return self.envs.render()
    
    def get_unit_masks(self):
        return np.array(self.envs.vec_client.getUnitLocationMasks())
    
    def get_unit_action_masks(self, units:np.ndarray):
        return np.array(self.envs.vec_client.getUnitActionMasks(units)).reshape(len(units), -1)
    
    def sample_action(self, distris: list[MaskedCategorical]):
        unit_masks = torch.Tensor(self.get_unit_masks()).reshape(self.num_env, -1)
        distris[0].update_masks(unit_masks)
        
        units = distris[0].sample()
        action_components = [units]

        action_mask_list = np.array(self.envs.vec_client.getUnitActionMasks(units.cpu().numpy())).reshape(len(units), -1)
        action_masks = torch.split(torch.Tensor(action_mask_list), self.action_space[1:], dim=1) 
        
        action_components +=  [dist.update_masks(action_mask).sample() for dist , action_mask in zip(distris[1:],action_masks)]
            
        actions = torch.stack(action_components)
        masks = torch.cat((unit_masks, torch.Tensor(action_mask_list)), 1)
        log_probs = torch.stack([dist.log_prob(aciton) for dist,aciton in zip(distris,actions)])
        
        return actions.T.cpu().numpy(), masks.cpu().numpy(),log_probs.T.cpu().numpy()
    
if __name__ == "__main__":
    num_env = 32
    test_env = EnvMicroRTSW(num_env=num_env,
                            max_steps=5000,
                            ais=[microrts_ai.coacAI for _ in range(num_env)],
                            map_path="maps/16x16/basesWorkers16x16.xml",
                            reward_weights=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]))
    obs = test_env.reset()
    
    random_actions_units = torch.ones((num_env,16*16))
    random_action_types = torch.ones((num_env,6))
    random_action_move = torch.ones((num_env,4))
    random_action_harvest = torch.ones((num_env,4))
    random_action_return = torch.ones((num_env,4))
    random_action_produce = torch.ones((num_env,4))
    random_action_produce_type = torch.ones((num_env,7))
    random_action_attack = torch.ones((num_env,49))

    distri = [MaskedCategorical(random_actions_units),
                     MaskedCategorical(random_action_types),
                     MaskedCategorical(random_action_move),
                     MaskedCategorical(random_action_harvest),
                     MaskedCategorical(random_action_return),
                     MaskedCategorical(random_action_produce),
                     MaskedCategorical(random_action_produce_type),
                     MaskedCategorical(random_action_attack)]
    
    for _ in range(100000):
        test_env.render()
        actions, masks, log_probs = test_env.sample_action(distri)
        obs, rs, done_n, infos = test_env.step(actions)
        

    
