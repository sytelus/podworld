from boxworld.envs import BoxWorldEnv
import time 

env = BoxWorldEnv()
env.reset()

env_done = False
start_time = time.time()
step_count = 0

while not env_done:
    obs, reward, env_done, info = env.step(action=None)
    step_count += 1

    rendered=env.render(mode='human')
    # print(obs)

print('Time, Step Count', time.time()-start_time, step_count)

env.close()
