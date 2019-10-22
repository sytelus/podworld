from podworld.envs import PodWorldEnv
import tensorwatch as tw
import time

start_time = time.time()
print('Started', start_time)

watcher = tw.Watcher(filename='random_agent.log')
logger = watcher.create_stream(name='reward')
watcher.make_notebook()

env = PodWorldEnv()
env.reset()

env_done, i, total_r = False, 0, 0.0
while not env_done and i < 10000:
    action = env.action_space.sample()
    obs, reward, env_done, info = env.step(action=action)
    #rendered=env.render(mode='human')
    total_r += reward
    logger.write((i, total_r))
    i += 1
    

print('Done.', time.time() - start_time)

