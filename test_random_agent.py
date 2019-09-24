from boxworld.envs import BoxWorldEnv
import time 

env = BoxWorldEnv()
env.reset()

env_done = False
start_time = time.time()

while not env_done:
    action = env.action_space.sample()
    obs, reward, env_done, info = env.step(action=action)
    rendered=env.render(mode='human')

print('Time, Step Count', time.time()-start_time, env.step_count)

env.close()
