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
import random

import wandb
import argparse

lr = 1e-4
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
max_clip_range = 4
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
cuda = True
device = 'cuda'
pae_length = 256
sample_length = 512
num_envs = 32
num_steps = 256

ai_dict = {
    "coacAI": microrts_ai.coacAI,
    "rojo": microrts_ai.rojo,
    "mayari": microrts_ai.mayari,
    "randomAI": microrts_ai.randomAI,
    "passiveAI": microrts_ai.passiveAI,
    "workerRushAI": microrts_ai.workerRushAI,
    "lightRushAI": microrts_ai.lightRushAI,
}

parser = argparse.ArgumentParser(description='BC')
parser.add_argument('--map_name', type=str, default='basesWorkers12x12')
parser.add_argument('--weight', type=int, default=10000)
parser.add_argument('--base_ai', type=str, default='coacAI')
parser.add_argument('--op_ai', type=str, default='coacAI')

op_ai = ai_dict[argparse.op_ai]
base_ai = ai_dict[argparse.base_ai]

#main network
class Actor(nn.Module):
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
        
    def get_distris(self,states):
        states = states.permute((0, 3, 1, 2))
        policy_network = self.policy_network(states)
        action_dist = self.actor(policy_network)
        return action_dist

    
    def forward(self, states):
        distris = self.get_distris(states)
        return distris
    
class Agent:
    def __init__(self,net:Actor) -> None:
        self.net = net
        self.num_envs = num_envs
        self.num_steps = num_steps
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
        base_dist = base_dist*argparse.weight
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
        masks = torch.cat((unit_masks, torch.Tensor(action_mask_list)), 1)
        log_probs = torch.stack([dist.log_prob(aciton) for dist,aciton in zip(distris,actions)])
        
        return actions.T.cpu().numpy(), masks.cpu().numpy(),log_probs.T.cpu().numpy(), base_dist
    
    def sample_bc_env(self, check=False):  
        if check:
           step_record_dict = dict()
           rewards = []
           log_probs = [] 
        for _ in range(self.num_steps):
            unit_mask = self.env.get_unit_masks().reshape(self.num_envs, -1)
            action,mask,log_prob,base_dist=self.get_sample_actions(self.obs, unit_mask)
            next_obs, rs, done_n, infos = self.env.step(action)
        
            if check:
                rewards.append(np.mean(rs))
                log_probs.append(np.mean(log_prob))
            
            for i in range(self.num_envs):
                self.exps_list[i].append([self.obs[i],base_dist[i]])
                if check:
                    if done_n[i]:
                        if infos[i]['raw_rewards'][0] > 0:
                            self.out_comes.append(1.0)
                        else:
                            self.out_comes.append(0.0)
                
            self.obs=next_obs

        if len(self.exps_list[0])>sample_length:
            train_exps = [ exps[-sample_length:] for exps in self.exps_list ]
            self.exps_list = train_exps
        else:
            train_exps = self.exps_list

        if check:
            mean_win_rates = np.mean(self.out_comes) if len(self.out_comes)>0 else 0.0
            print(mean_win_rates)

            self.rewards.append(np.mean(rewards))
            step_record_dict['mean_rewards'] = np.mean(self.rewards)
            step_record_dict['mean_log_probs'] = np.mean(log_probs)
            step_record_dict['mean_win_rates'] = mean_win_rates
            return train_exps, step_record_dict
        
        return train_exps
    
    def sample_base_actions(self):
        for _ in range(self.num_steps):
            unit_mask = self.env.get_unit_masks().reshape(self.num_envs, -1)
            action,mask,log_prob,base_dist=self.get_sample_actions(self.obs, unit_mask)
            next_obs, rs, done_n, infos = self.env.step(action)
            self.obs=next_obs

class BCCalculator:
    def __init__(self,net:Actor) -> None:
        self.net = net
        self.train_version = 0
        
        if cuda and torch.cuda.is_available():
            self.device = torch.device('cuda', 0)
        else:
            self.device = torch.device('cpu')
        
        self.calculate_net = Actor()
        self.calculate_net.to(self.device)
    
        self.share_optim = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        
        
        self.states_list = None
        self.base_dists_list = None

    def begin_batch_train(self, samples_list: list):    
        s_states = [np.array([s[0] for s in samples]) for samples in samples_list]
        s_base_dists = [np.array([s[1] for s in samples]) for samples in samples_list]
        
        self.states = [torch.Tensor(states).to(self.device) for states in s_states]
        self.base_dists = [torch.Tensor(base_dists).to(self.device) for base_dists in s_base_dists]
        
        self.states_list = torch.cat(self.states)
        self.base_dists_list = torch.cat(self.base_dists)
        
    def end_batch_train(self):
        self.states_list = None
        self.base_dists_list = None


    def generate_grads(self):
        grad_norm = max_grad_norm
        
        self.calculate_net.load_state_dict(self.net.state_dict())

        mini_batch_number = 1
        mini_batch_size = self.states_list.shape[0]

        for i in range(mini_batch_number):
            start_index = i*mini_batch_size
            end_index = (i+1)* mini_batch_size
            
            mini_states = self.states_list[start_index:end_index]
            mini_base_dists = self.base_dists_list[start_index:end_index]
            
            self.calculate_net.load_state_dict(self.net.state_dict())
                
            mini_dists = self.calculate_net.get_distris(mini_states)

            mini_base_dists = mini_base_dists.split(action_space, dim=1)
            mini_base_dists = [F.softmax(dist,dim=-1) for dist in mini_base_dists]
            mini_base_dists = torch.cat(mini_base_dists,dim=1)

            mini_dists = mini_dists.split(action_space, dim=1)
            mini_dists = [F.softmax(dist,dim=-1) for dist in mini_dists]
            mini_dists = torch.cat(mini_dists,dim=1)


            loss = F.mse_loss(mini_dists,mini_base_dists)

            self.calculate_net.zero_grad()

            loss.backward()
            
            grads = [
                param.grad.data.cpu().numpy()
                if param.grad is not None else None
                for param in self.calculate_net.parameters()
            ]
                
            # Updating network parameters
            for param, grad in zip(self.net.parameters(), grads):
                if grad is not None:
                    param.grad = torch.FloatTensor(grad)
                
            if grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),grad_norm)
            self.share_optim.step()

if __name__ == "__main__":
    map_name = argparse.map_name
    seed = random.randint(0,100000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
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
    comment = "bc_"+map_name+"_"+str(seed)
    net = Actor()
    parameters = sum([np.prod(p.shape) for p in net.parameters()])
    print("parameters size is:",parameters)

    agent = Agent(net)
    bc_calculator = BCCalculator(net)

    MAX_VERSION = 500
    REPEAT_TIMES = 10

    wandb.init(project="MicrortsImitationLearning", name=comment, config={
        "base_ai": argparse.base_ai,
        "op_ai": argparse.op_ai,
        "map": map_name,
        "epochs": MAX_VERSION,
        "samples_per_epochs":num_envs*pae_length,
        "start_win_rate":0.0,
        "temperature_coefficient":-2,
        })

    for version in range(MAX_VERSION):
        samples_list,infos = agent.sample_bc_env(check=True)

        infos["global_steps"] = version*num_envs*sample_length
        wandb.log(infos,step=version)

        mean_win_rates = infos["mean_win_rates"]

        print("version:",version,"reward:",infos["mean_rewards"])

        samples = []

        for s in samples_list:
            samples.append(s)
        
        bc_calculator.begin_batch_train(samples)
        for _ in range(REPEAT_TIMES):
            bc_calculator.generate_grads()
        bc_calculator.end_batch_train()

    try:
        torch.save(net.state_dict(), "saved_model/bc_agent/"+comment+".pt")
    except:
        print("save model error")
    wandb.finish()