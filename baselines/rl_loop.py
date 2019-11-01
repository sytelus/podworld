import tensorwatch as tw
import time
from base_agent import BaseAgent
import gym

def run_episode(env:gym.Env, agent:BaseAgent, render=False):
    start_time = time.time()
    print('Started', start_time)

    watcher = tw.Watcher(filename='random_agent.log')
    logger = watcher.create_stream(name='reward')
    watcher.make_notebook()

    obs = env.reset()
    agent.reset(env)

    reward, env_done, i, total_r = 0.0, False, 0, 0.0
    while not env_done:
        action = agent.act(obs, reward, env_done)
        obs, reward, env_done, info = env.step(action=action)
        if render:
            rendered = env.render(mode='human')
        total_r += reward
        logger.write((i, total_r))
        i += 1

    print('Done: reward, time', total_r, time.time() - start_time)
    return total_r

