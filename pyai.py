from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
import numpy as np
from utils import MaskedCategorical
import torch
import time

class pyMicorRTSAI:
    def __init__(self, num_env, utt, h,w,ai=microrts_ai.coacAI) -> None:
        self.num_env = num_env
        self.my_ai = ai(utt)
        self.h = h
        self.w = w
        self.size = h*w
        self.action_space = [self.size, 6, 4, 4, 4, 4, 7, 7*7]
    
    def getSampleAction(self, player, gss, env):
        actions_disrtir = np.zeros((self.num_env, self.size+78), dtype=np.int32)
        units_disrtir = np.array(env.vec_client.getUnitLocationMasks()).reshape(self.num_env, -1)
        units = np.zeros(self.num_env, dtype=np.int32)
        for i in range(self.num_env):
            res_action = np.zeros(self.size+78, dtype=np.int32)

            try:
                gs = gss[i]
                ac = self.my_ai.getAction(player,gs)
                aac = ac.getActions()
                p = None
                if len(aac) == 0:
                    continue
                for j in range(len(aac)):
                    u = aac[j].m_a
                    u_pos_x = u.getX()
                    u_pos_y = u.getY()
                    u_pos = u_pos_x + u_pos_y*self.w
                    if units_disrtir[i][u_pos] == 1:
                        p = aac[j]
                        break
                
                if p is None:
                    continue
                
                # p = ac.getActions()
                unit = p.m_a
                unit_action = p.m_b
                
                unit_pos_x = unit.getX()
                unit_pos_y = unit.getY()

                unit_pos = unit_pos_x + unit_pos_y*self.w
                res_action[unit_pos] = 1
                units[i] = unit_pos

                unit_action_type = unit_action.getActionName()
                if unit_action_type == "move":
                    res_action[self.size+1] = 1
                elif unit_action_type == "harvest":
                    res_action[self.size+2] = 1
                elif unit_action_type == "return":
                    res_action[self.size+3] = 1
                elif unit_action_type == "produce":
                    res_action[self.size+4] = 1
                elif unit_action_type == "attack":
                    res_action[self.size+5] = 1

                direction = unit_action.getDirection()
                if unit_action_type == "move":
                    res_action[self.size+6+direction] = 1
                elif unit_action_type == "harvest":
                    res_action[self.size+6+4+direction] = 1
                elif unit_action_type == "return":
                    res_action[self.size+6+4+4+direction] = 1
                elif unit_action_type == "produce":
                    res_action[self.size+6+4+4+4+direction] = 1

                if unit_action.getUnitType() is not None:
                    produce_unit = unit_action.getUnitType().name
                    if produce_unit == "base":
                        res_action[self.size+6+4+4+4+4+1] = 1
                    elif produce_unit == "barracks":
                        res_action[self.size+6+4+4+4+4+2] = 1
                    elif produce_unit == "worker":
                        res_action[self.size+6+4+4+4+4+3] = 1
                    elif produce_unit == "light":
                        res_action[self.size+6+4+4+4+4+4] = 1
                    elif produce_unit == "heavy":
                        res_action[self.size+6+4+4+4+4+5] = 1
                    elif produce_unit == "ranged":
                        res_action[self.size+6+4+4+4+4+6] = 1
                else:
                    produce_unit = None
                    res_action[self.size+6+4+4+4+4] = 1

                if unit_action_type == "attack":
                    target_x = unit_action.getLocationX()
                    target_y = unit_action.getLocationY()
                    r_x = target_x - unit_pos_x+3
                    r_y = target_y - unit_pos_y+3
                    if 0<=r_x<=6 and 0<=r_y<=6:
                        attack_pos = r_x + r_y*7
                        res_action[self.size+6+4+4+4+4+7+attack_pos] = 1

                actions_disrtir[i] = res_action
            except:
                actions_disrtir[i] = res_action

        return actions_disrtir, units
    
    def getExploreAction(self, player, gss, env):
        actions_disrtir = np.zeros((self.num_env, self.size+78), dtype=np.int32)
        units_disrtir = np.array(env.vec_client.getUnitLocationMasks()).reshape(self.num_env, -1)
        units = np.zeros(self.num_env, dtype=np.int32)
        for i in range(self.num_env):
            res_action = np.zeros(self.size+78, dtype=np.int32)
            gs = gss[i]
            ac = self.my_ai.getAction(player,gs)
            aac = ac.getActions()
            p = None
            if len(aac) == 0:
                continue
            for j in range(len(aac)):
                u = aac[j].m_a
                u_pos_x = u.getX()
                u_pos_y = u.getY()
                u_pos = u_pos_x + u_pos_y*self.w
                if units_disrtir[i][u_pos] == 1:
                    p = aac[j]
                    break
            
            if p is None:
                continue
            
            # p = ac.getActions()
            unit = p.m_a
            unit_action = p.m_b
            
            unit_pos_x = unit.getX()
            unit_pos_y = unit.getY()

            unit_pos = unit_pos_x + unit_pos_y*self.w
            res_action[unit_pos] = 1
            units[i] = unit_pos

            unit_action_type = unit_action.getActionName()
            if unit_action_type == "move":
                res_action[self.size+1] = 1
            elif unit_action_type == "harvest":
                res_action[self.size+2] = 1
            elif unit_action_type == "return":
                res_action[self.size+3] = 1
            elif unit_action_type == "produce":
                res_action[self.size+4] = 1
            elif unit_action_type == "attack":
                res_action[self.size+5] = 1

            direction = unit_action.getDirection()
            if unit_action_type == "move":
                res_action[self.size+6+direction] = 1
            elif unit_action_type == "harvest":
                res_action[self.size+6+4+direction] = 1
            elif unit_action_type == "return":
                res_action[self.size+6+4+4+direction] = 1
            elif unit_action_type == "produce":
                res_action[self.size+6+4+4+4+direction] = 1

            actions_disrtir[i] = res_action

        return actions_disrtir, units
            
    def getAction(self, player, gss, env):
        actions = np.zeros((self.num_env, 8), dtype=np.int32)
        units_disrtir = np.array(env.vec_client.getUnitLocationMasks()).reshape(self.num_env, -1)
        for i in range(self.num_env):
            res_action = np.zeros(8, dtype=np.int32)
            gs = gss[i]
            ac = self.my_ai.getAction(player,gs)
            aac = ac.getActions()
            p = None
            if len(aac) == 0:
                continue
            for j in range(len(aac)):
                u = aac[j].m_a
                u_pos_x = u.getX()
                u_pos_y = u.getY()
                u_pos = u_pos_x + u_pos_y*self.w
                if units_disrtir[i][u_pos] == 1:
                    p = aac[j]
                    break
            
            if p is None:
                continue
            
            # p = ac.getActions()
            unit = p.m_a
            unit_action = p.m_b
            
            unit_pos_x = unit.getX()
            unit_pos_y = unit.getY()

            unit_pos = unit_pos_x + unit_pos_y*self.w
            res_action[0] = unit_pos

            unit_action_type = unit_action.getActionName()
            if unit_action_type == "move":
                res_action[1] = 1
            elif unit_action_type == "harvest":
                res_action[1] = 2
            elif unit_action_type == "return":
                res_action[1] = 3
            elif unit_action_type == "produce":
                res_action[1] = 4
            elif unit_action_type == "attack_location":
                res_action[1] = 5

            direction = unit_action.getDirection()
            if unit_action_type == "move":
                res_action[2] = direction
            elif unit_action_type == "harvest":
                res_action[3] = direction
            elif unit_action_type == "return":
                res_action[4] = direction
            elif unit_action_type == "produce":
                res_action[5] = direction

            if unit_action.getUnitType() is not None:
                produce_unit = unit_action.getUnitType().name
                if produce_unit == "Base":
                    res_action[6] = 1
                elif produce_unit == "Barracks":
                    res_action[6] = 2
                elif produce_unit == "Worker":
                    res_action[6] = 3
                elif produce_unit == "Light":
                    res_action[6] = 4
                elif produce_unit == "Heavy":
                    res_action[6] = 5
                elif produce_unit == "Ranged":
                    res_action[6] = 6
            else:
                produce_unit = None
                res_action[6] = 0

            if unit_action_type == "attack_location":
                target_x = unit_action.getLocationX()
                target_y = unit_action.getLocationY()
                r_x = target_x - unit_pos_x+3
                r_y = target_y - unit_pos_y+3
                if 0<=r_x<=6 and 0<=r_y<=6:
                    attack_pos = r_x + r_y*7
                    res_action[7] = attack_pos

            actions[i] = res_action
        
        return actions
        
if __name__ == "__main__":
    map_name = "basesWorkers16x16"
    h=16
    w=16
    map_path = "maps/16x16/basesWorkers16x16.xml"
    if map_name == "DoubleGame24x24":
        map_path = "maps/DoubleGame24x24.xml"
        h=24
        w=24
    elif map_name == "basesWorkers16x16noResources":
        map_path = "maps/16x16/basesWorkers16x16NoResources.xml"
        h=16
        w=16
    elif map_name == "basesWorkers24x24":
        map_path = "maps/24x24/basesWorkers24x24.xml"
        h=24
        w=24
    elif map_name == "basesWorkers24x24L":
        map_path = "maps/24x24/basesWorkers24x24L.xml"
        h=24
        w=24
    elif map_name == "basesWorkers12x12":
        map_path = "maps/12x12/basesWorkers12x12.xml"
        h=12
        w=12
    elif map_name == "FourBasesWorkers12x12":
        map_path = "maps/12x12/FourBasesWorkers12x12.xml"
        h=12
        w=12
    elif map_name == "TwoBasesBarracks16x16":
        map_path = "maps/16x16/TwoBasesBarracks16x16.xml"
        h=16
        w=16


    num_envs = 16
    env = MicroRTSVecEnv(
        num_envs=num_envs,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.coacAI for _ in range(num_envs)],
        map_path=map_path,
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    )
    obs = env.reset()

    outcomes = []

    my_ai = pyMicorRTSAI(num_envs, env.real_utt, w,h)

    for _ in range(100000):
        env.render()
        
        gss = []
        for i in range(num_envs):
            gss += [env.vec_client.clients[i].gs]
        #actions, units = my_ai.getSampleAction(0, gss, env)
        actions = my_ai.getAction(0, gss, env)
        actions = actions.astype(np.int64)
        obs, r, done, info = env.step(actions)

        for j in range(num_envs):
            if done[j]:
                r_win = info[j]["raw_rewards"][0]
                if r_win > 0:
                    outcomes += [1]
                else:
                    outcomes += [0]
                print("Episode: {}, Outcome: {}".format(len(outcomes), sum(outcomes)))
