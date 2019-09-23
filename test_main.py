from boxworld.envs import BoxWorldEnv

env = BoxWorldEnv()
env.reset()

env_done = False
while not env_done:
    obs, reward, env_done, info = env.step(action=None)
    rendered=env.render(mode='human')

env.close()
