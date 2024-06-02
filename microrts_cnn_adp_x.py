import sys,os,time
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import torch 
import torch.nn as nn
from env_microrts import EnvMicroRTSW
from gym_microrts import microrts_ai
from collections import deque
import numpy as np
import torch.nn.functional as F

from utils import layer_init, calculate_gae, MaskedCategorical

from pyai import pyMicorRTSAI

import wandb
import random

import argparse

lr = 2.5e-4
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
max_clip_range = 4
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
cuda = True
device = 'cuda'
pae_length = 128
num_envs = 32
num_steps = 512

ai_dict = {
    "coacAI": microrts_ai.coacAI,
    "rojo": microrts_ai.rojo,
    "mayari": microrts_ai.mayari,
    "randomAI": microrts_ai.randomAI,
    "passiveAI": microrts_ai.passiveAI,
    "workerRushAI": microrts_ai.workerRushAI,
    "lightRushAI": microrts_ai.lightRushAI,
}


parser = argparse.ArgumentParser(description='AdapterX-RL')
parser.add_argument('--map_name', type=str, default='basesWorkers12x12')
parser.add_argument('--weight', type=int, default=100)
parser.add_argument('--base_ai', type=str, default='coacAI')
parser.add_argument('--op_ai', type=str, default='coacAI')

args = parser.parse_args()

base_ai = ai_dict[args.base_ai]
op_ai = ai_dict[args.op_ai]

#main network
class ActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy_network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(cnn_output_dim, 256)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(256, sum(action_space)))
        
        self.value = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(cnn_output_dim, 256)),
                nn.ReLU(), 
                layer_init(nn.Linear(256, 1), std=1)
            )
        
    def get_distris(self,states):
        states = states.permute((0, 3, 1, 2))
        policy_network = self.policy_network(states)
        action_dist = self.actor(policy_network)
        return action_dist

    def get_value(self, states):
        states = states.permute((0, 3, 1, 2))
        value = self.value(states)
        return value
    
    def forward(self, states):
        distris = self.get_distris(states)
        value = self.get_value(states)
        return distris,value

class Agent:
    def __init__(self,net:ActorCritic) -> None:
        self.net = net
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.pae_length = pae_length
        self.action_space = action_space
        self.out_comes = deque( maxlen= 100)
        self.rewards = deque( maxlen= 100)
        self.env = EnvMicroRTSW(num_env=num_envs,
                            max_steps=20000,
                            ais=[op_ai for _ in range(num_envs)],
                            map_path=map_path,
                            reward_weights=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]))
        self.obs = self.env.reset()
        self.exps_list = [[] for _ in range(self.num_envs)]
        utt = self.env.envs.real_utt
        self.base_ai = pyMicorRTSAI(self.num_envs, utt, w, h, ai=base_ai)
    
    @torch.no_grad()
    def get_sample_actions(self,states, unit_masks):
        gss = []
        for i in range(num_envs):
            gss += [self.env.envs.vec_client.clients[i].gs]
        base_dist, _ = self.base_ai.getSampleAction(0, gss, self.env.envs)
        states = torch.Tensor(states)
        distris = self.net.get_distris(states)
        
        # base_dist = torch.zeros_like(distris)

        base_dist = base_dist*args.weight
        base_dist = torch.tensor(base_dist, dtype=torch.float32)
        base_dist = base_dist.split(self.action_space, dim=1)
        base_dist = [F.softmax(dist, dim=-1) for dist in base_dist]
        base_dist = torch.cat(base_dist, dim=1)

        #distris = distris*base_dist
        distris = torch.mul(distris, base_dist)

        distris = distris.split(self.action_space, dim=1)
        distris = [MaskedCategorical(dist) for dist in distris]
        
        unit_masks = torch.Tensor(unit_masks)
        distris[0].update_masks(unit_masks)

        distris_ = distris
        
        units = distris_[0].sample()
        action_components = [units]

        action_mask_list = self.env.get_unit_action_masks(units.cpu().numpy())
        action_masks = torch.split(torch.Tensor(action_mask_list), self.action_space[1:], dim=1) 
        
        action_components +=  [dist.update_masks(action_mask).sample() for dist , action_mask in zip(distris_[1:],action_masks)]
            
        actions = torch.stack(action_components)
        masks = torch.cat((unit_masks, torch.Tensor(action_mask_list)), 1)
        log_probs = torch.stack([dist.log_prob(aciton) for dist,aciton in zip(distris_,actions)])
        adp_log_probs = torch.stack([dist.log_prob(aciton) for dist,aciton in zip(distris,actions)])
        
        return actions.T.cpu().numpy(), masks.cpu().numpy(),log_probs.T.cpu().numpy(), base_dist.cpu().numpy(), adp_log_probs.T.cpu().numpy()
    
    def sample_env(self, check=False):  
        if check:
           step_record_dict = dict()
           rewards = []
           log_probs = []
           adp_log_probs = [] 
        while len(self.exps_list[0]) < self.num_steps:
            unit_mask = self.env.get_unit_masks().reshape(self.num_envs, -1)
            action,mask,log_prob,base_dist,adp_log_prob =self.get_sample_actions(self.obs, unit_mask)
            next_obs, rs, done_n, infos = self.env.step(action)
        
            if check:
                rewards.append(np.mean(rs))
                log_probs.append(np.mean(log_prob))
                adp_log_probs.append(np.mean(adp_log_prob))
            
            for i in range(self.num_envs):
                if done_n[i]:
                    done = True
                else:
                    done = False
                self.exps_list[i].append([self.obs[i],action[i],rs[i],mask[i],done,log_prob[i],base_dist[i]])
                if check:
                    if done_n[i]:
                        if infos[i]['raw_rewards'][0] > 0:
                            self.out_comes.append(1.0)
                        else:
                            self.out_comes.append(0.0)
                
            self.obs=next_obs

        train_exps = self.exps_list
        self.exps_list = [ exps[self.pae_length:self.num_steps] for exps in self.exps_list ]

        if check:
            mean_win_rates = np.mean(self.out_comes) if len(self.out_comes)>0 else 0.0
            print(mean_win_rates)

            self.rewards.append(np.mean(rewards))
            step_record_dict['mean_rewards'] = np.mean(self.rewards)
            step_record_dict['mean_log_probs'] = np.mean(log_probs)
            step_record_dict['mean_win_rates'] = mean_win_rates
            step_record_dict['adp_log_probs'] = np.mean(adp_log_probs)
            return train_exps, step_record_dict
        
        return train_exps
    
    def get_action_without_base_agent(self,states, unit_masks):
        states = torch.Tensor(states)
        distris = self.net.get_distris(states)
        distris = distris.split(self.action_space, dim=1)
        distris = [MaskedCategorical(dist) for dist in distris]
        
        unit_masks = torch.Tensor(unit_masks)
        distris[0].update_masks(unit_masks)
        
        units = distris[0].sample()
        action_components = [units]

        action_mask_list = self.env.get_unit_action_masks(units.cpu().numpy())
        action_masks = torch.split(torch.Tensor(action_mask_list), self.action_space[1:], dim=1) 
        
        action_components +=  [dist.update_masks(action_mask).sample() for dist , action_mask in zip(distris[1:],action_masks)]
            
        actions = torch.stack(action_components)
        
        return actions.T.cpu().numpy()
    
    def check(self):
        self.obs = self.env.reset()
        step_record_dict = dict()
        rewards = []
        out_comes = []
        while True:
            unit_mask = self.env.get_unit_masks().reshape(self.num_envs, -1)
            action = self.get_action_without_base_agent(self.obs, unit_mask)
            next_obs, rs, done_n, infos = self.env.step(action)
        
            rewards.append(np.mean(rs))
            
            for i in range(self.num_envs):
                if done_n[i]:
                    done = True
                else:
                    done = False
                if done:
                    if infos[i]['raw_rewards'][0] > 0:
                        out_comes.append(1.0)
                    else:
                        out_comes.append(0.0)
            
            self.obs=next_obs

            if len(out_comes) >= 100:
                break
        win_rate = np.mean(out_comes)
        step_record_dict['mean_rewards'] = np.mean(rewards)
        step_record_dict['only_adpter_mean_win_rates'] = win_rate
        return step_record_dict


class Calculator:
    def __init__(self,net:ActorCritic) -> None:
        self.net = net
        self.train_version = 0
        self.pae_length = pae_length
        
        if cuda and torch.cuda.is_available():
            self.device = torch.device('cuda', 0)
        else:
            self.device = torch.device('cpu')
        
        self.calculate_net = ActorCritic()
        self.calculate_net.to(self.device)
    
        self.share_optim = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        
        
        self.states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.old_log_probs_list = None
        self.marks_list = None
        self.base_dists_list = None

    def begin_batch_train(self, samples_list: list):    
        s_states = [np.array([s[0] for s in samples]) for samples in samples_list]
        s_actions = [np.array([s[1] for s in samples]) for samples in samples_list]
        s_masks = [np.array([s[3] for s in samples]) for samples in samples_list]
        s_log_probs = [np.array([s[5] for s in samples]) for samples in samples_list]
        s_base_dists = [np.array([s[6] for s in samples]) for samples in samples_list]
        
        s_rewards = [np.array([s[2] for s in samples]) for samples in samples_list]
        s_dones = [np.array([s[4] for s in samples]) for samples in samples_list]
        
        self.states = [torch.Tensor(states).to(self.device) for states in s_states]
        self.actions = [torch.Tensor(actions).to(self.device) for actions in s_actions]
        self.old_log_probs = [torch.Tensor(log_probs).to(self.device) for log_probs in s_log_probs]
        self.marks = [torch.Tensor(marks).to(self.device) for marks in s_masks]
        self.base_dists = [torch.Tensor(base_dists).to(self.device) for base_dists in s_base_dists]
        self.rewards = s_rewards
        self.dones = s_dones
        
        self.states_list = torch.cat([states[0:self.pae_length] for states in self.states])
        self.actions_list = torch.cat([actions[0:self.pae_length] for actions in self.actions])
        self.old_log_probs_list = torch.cat([old_log_probs[0:self.pae_length] for old_log_probs in self.old_log_probs])
        self.marks_list = torch.cat([marks[0:self.pae_length] for marks in self.marks])
        self.base_dists_list = torch.cat([base_dists[0:self.pae_length] for base_dists in self.base_dists])

    def calculate_samples_gae(self):
        np_advantages = []
        np_returns = []
        
        for states,rewards,dones in zip(self.states,self.rewards,self.dones):
            with torch.no_grad():
                values = self.calculate_net.get_value(states)
                            
            advantages,returns = calculate_gae(values.cpu().numpy().reshape(-1),rewards,dones,gamma,gae_lambda)
            np_advantages.extend(advantages[0:self.pae_length])
            np_returns.extend(returns[0:self.pae_length])
            
        np_advantages = np.array(np_advantages)
        np_returns = np.array(np_returns)
        
        return np_advantages, np_returns
        
    def end_batch_train(self):
        self.states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.old_log_probs_list = None
        self.marks_list = None
        self.base_dists_list = None

    def get_pg_loss(self,ratio,advantage):      
        clip_coef = clip_range
        max_clip_coef = max_clip_range
        positive = torch.where(ratio >= 1.0 + clip_coef, 0 * advantage,advantage)
        negtive = torch.where(ratio <= 1.0 - clip_coef,0 * advantage,torch.where(ratio >= max_clip_coef, 0 * advantage,advantage))
        return torch.where(advantage>=0,positive,negtive)*ratio
        
    def get_prob_entropy_value(self,states, actions, masks, base_dists):
        distris = self.calculate_net.get_distris(states)
        distris = torch.mul(distris, base_dists)
        distris = distris.split(action_space, dim=1)
        distris = [MaskedCategorical(dist) for dist in distris]
        values = self.calculate_net.get_value(states)
        action_masks = torch.split(masks, action_space, dim=1)
        distris = [dist.update_masks(mask,device=self.device) for dist,mask in zip(distris,action_masks)]
        log_probs = torch.stack([dist.log_prob(action) for dist,action in zip(distris,actions)])
        entropys = torch.stack([dist.entropy() for dist in distris])
        return log_probs.T, entropys.T, values

    def generate_grads(self):
        grad_norm = max_grad_norm
        
        self.calculate_net.load_state_dict(self.net.state_dict())
        np_advantages,np_returns = self.calculate_samples_gae()
        
        np_advantages = (np_advantages - np_advantages.mean()) / np_advantages.std()
                                                    
        advantage_list = torch.Tensor(np_advantages.reshape(-1,1)).to(self.device)    
        returns_list = torch.Tensor(np_returns.reshape(-1,1)).to(self.device)
        

        mini_batch_number = 1
        mini_batch_size = advantage_list.shape[0]

        for i in range(mini_batch_number):
            start_index = i*mini_batch_size
            end_index = (i+1)* mini_batch_size
            
            mini_states = self.states_list[start_index:end_index]
            mini_actions = self.actions_list[start_index:end_index]
            mini_masks = self.marks_list[start_index:end_index]
            mini_base_dists = self.base_dists_list[start_index:end_index]
            mini_old_log_probs = self.old_log_probs_list[start_index:end_index]
            
            self.calculate_net.load_state_dict(self.net.state_dict())
                
            mini_new_log_probs,mini_entropys,mini_new_values = self.get_prob_entropy_value(mini_states,mini_actions.T,mini_masks,mini_base_dists)
                        
            mini_advantage = advantage_list[start_index:end_index]
            mini_returns = returns_list[start_index:end_index]
            
            ratio1 = torch.exp(mini_new_log_probs-mini_old_log_probs)
            pg_loss = self.get_pg_loss(ratio1,mini_advantage)

            # Policy loss
            pg_loss = -torch.mean(pg_loss)
            
            entropy_loss = -torch.mean(mini_entropys)
            
            v_loss = F.mse_loss(mini_new_values, mini_returns)

            loss = pg_loss+ v_loss*vf_coef + ent_coef * entropy_loss 

            self.calculate_net.zero_grad()

            loss.backward()
            
            grads = [
                param.grad.data.cpu().numpy()
                if param.grad is not None else None
                for param in self.calculate_net.parameters()
            ]
                
            # Updating network parameters
            for param, grad in zip(self.net.parameters(), grads):
                param.grad = torch.FloatTensor(grad)
                
            if grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),grad_norm)
            self.share_optim.step()


if __name__ == "__main__":
    MAX_VERSION = 500
    REPEAT_TIMES = 10
    map_name = args.map_name
    if map_name == "basesWorkers16x16noResources":
        map_path = "maps/16x16/basesWorkers16x16NoResources.xml"
        h=16
        w=16
        cnn_output_dim = 32*6*6
    elif map_name == "basesWorkers12x12":
        map_path = "maps/12x12/basesWorkers12x12.xml"
        h=12
        w=12
        cnn_output_dim = 32*4*4
    elif map_name == "TwoBasesBarracks16x16":
        map_path = "maps/16x16/TwoBasesBarracks16x16.xml"
        h=16
        w=16
        cnn_output_dim = 32*6*6

    action_space = [w*h, 6, 4, 4, 4, 4, 7, 49]
    observation_space = [w,h,27]

    seed = random.randint(0,100000000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    global_steps = 0
    comment = map_name+"_"+str(args.weight)
    
    net = ActorCritic()
    parameters = sum([np.prod(p.shape) for p in net.parameters()])
    print("parameters size is:",parameters)

    agent = Agent(net)
    calculator = Calculator(net)

    while True:
        samples_list,infos = agent.sample_env(check=True)
        print("check game:", len(agent.out_comes))
        if len(agent.out_comes) >= 99:
            break
    
    start_win_rate = np.mean(agent.out_comes)
    wandb.init(
        # set the wandb project where this run will be logged
        project='AdapterX-RL-Project2',
        name = comment+str(seed),
        group = comment,

        # track hyperparameters and run metadata
        config={
        "base_ai": args.base_ai,
        "op_ai": args.op_ai,
        "map": map_name,
        "epochs": MAX_VERSION,
        "samples_per_epochs":num_envs*pae_length,
        "start_win_rate":start_win_rate,
        "temperature_coefficient":1/args.weight,
        }
    )

    for version in range(MAX_VERSION):
        global_steps = version*num_envs*pae_length
        samples_list,infos = agent.sample_env(check=True)

        infos["global_steps"] = global_steps
        wandb.log(infos)

        print("version:",version,"reward:",infos["mean_rewards"])

        samples = []

        for s in samples_list:
            samples.append(s)
        
        calculator.begin_batch_train(samples)
        for _ in range(REPEAT_TIMES):
            calculator.generate_grads()
        calculator.end_batch_train()
        
    wandb.finish()