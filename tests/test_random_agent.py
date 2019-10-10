from podworld.envs import PodWorldEnv
import time 

env = PodWorldEnv()
env.reset()

env_done = False
start_time = time.time()

while not env_done:
    action = env.action_space.sample()
    obs, reward, env_done, info = env.step(action=action)
    rendered=env.render(mode='human')

print('Time, Step Count', time.time()-start_time, env.step_count)

env.close()
