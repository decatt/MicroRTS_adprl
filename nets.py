import torch 
import torch.nn as nn
from utils import layer_init, calculate_gae, MaskedCategorical

class CNNNet(nn.Module):
    def __init__(self,cnn_output_dim,pos_output) -> None:
        super().__init__()
        self.policy_network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(cnn_output_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
        )

        self.policy_unit = layer_init(nn.Linear(256, pos_output), std=0.01)
        self.policy_type = layer_init(nn.Linear(256, 6), std=0.01)
        self.policy_move = layer_init(nn.Linear(256, 4), std=0.01)
        self.policy_harvest = layer_init(nn.Linear(256, 4), std=0.01)
        self.policy_return = layer_init(nn.Linear(256, 4), std=0.01)
        self.policy_produce = layer_init(nn.Linear(256, 4), std=0.01)
        self.policy_produce_type = layer_init(nn.Linear(256, 7), std=0.01)
        self.policy_attack = layer_init(nn.Linear(256, 49), std=0.01)
        
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
        unit_distris = MaskedCategorical(self.policy_unit(policy_network))
        type_distris = MaskedCategorical(self.policy_type(policy_network))
        move_distris = MaskedCategorical(self.policy_move(policy_network))
        harvest_distris = MaskedCategorical(self.policy_harvest(policy_network))
        return_distris = MaskedCategorical(self.policy_return(policy_network))
        produce_distris = MaskedCategorical(self.policy_produce(policy_network))
        produce_type_distris = MaskedCategorical(self.policy_produce_type(policy_network))
        attack_distris = MaskedCategorical(self.policy_attack(policy_network))

        return [unit_distris,type_distris,move_distris,harvest_distris,return_distris,produce_distris,produce_type_distris,attack_distris]

    def get_value(self, states):
        states = states.permute((0, 3, 1, 2))
        value = self.value(states)
        return value
    
    def forward(self, states):
        distris = self.get_distris(states)
        value = self.get_value(states)
        return distris,value