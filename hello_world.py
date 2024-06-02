import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai

env = MicroRTSVecEnv(
    num_envs=1,
    render_theme=2,
    ai2s=[microrts_ai.coacAI],
    map_path="maps/10x10/basesTwoWorkers10x10.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)

env.action_space.seed(0)
env.reset()
for i in range(1000000):
    env.render()
    actions = np.array([env.action_space.sample()],dtype=np.int64)
    next_obs, reward, done, info = env.step(actions)
    if done:
        env.reset()
env.close()