# Welcome to PodWorld

PodWorld is an [OpenAI Gym](https://gym.openai.com/) environment for [reinforcement learning](http://incompleteideas.net/book/the-book-2nd.html) experimentation. PodWorld is specifically designed to be partially observable and dynamic (hence abbreviation P.O.D.). We emphasize these two attributes to force agents to learn spatial as well as temporal representations that are more general than need in a fixed layout. In addition, all entities in PodWorld obey laws of physics allowing for long-tail of emergent behaviors that may not be available in games designed with hand-crafted rules for human entertainment. 

PodWorld is fast (>500 FPS on usual laptops) without needing GPU, is cross-platform and can run in headless mode. PodWorld is highly customizable and meant to be hackable so new task definitions can easily be accommodated and the difficulty level can easily be changed across several dimensions.

## How to Install

We highly recommend Anaconda to manage your Python environment.

### From Source
Installing from source is recommended (pypi package may not be latest).
```
git clone https://github.com/sytelus/podworld.git
cd podworld
pip install . -e
```

### Package install
```
pip install podworld
```

## How to Use

PodWord implements the full OpenAI Gym interface, making it easy to instantiate and use the environment. For example, the code snippet below implements a random agent:

```
from podworld.envs import PodWorldEnv

env = PodWorldEnv()
env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action=action)
    # comment below to run in headless mode
    rendered = env.render(mode='human')
```
[View source](tests/test_random_agent.py).

## Observation, Action and Reward
<img src="podworld.gif">

All objects in PodWord obey laws of physics, have mass and can move due to the application of force.

In the screenshot above, the agent is colored in red. The agent perceives the world through its 64-pixel 360-degree camera generating 1x64 RGB image of its view. The camera can see only up to N units of distance, giving rise to inherent partial observability. The top strip in the screenshot shows the agent's view of its surroundings. The black color indicates empty space.

The agent can move in the world using its 16 actuators symmetrically placed around it. The action space simply specifies the 1-based integer index of actuator that the agent wants to activate. When an actuator is activated, it generates an impulse of constant magnitude causing motion in the opposite direction. To activate no actuators select the action = 0. 

The environment has obstacles colored in purple that the agent should avoid colliding with. If the agent does collide with an obstacle, it receives a negative reward proportional to the momentum of the obstacle. Small obstacles moving fast and large obstacles moving slow hurt equally bad.

The environment provides the agent with food, colored in green. The agent should strive to collide with food as much as it can. When the agent collides with food, it receives a positive reward proportional to the mass of the food object. After collision, the food object turns grey, indicating that it has been eaten. After some random amount ot time, the food will become green again. When the food is grey, the agent receives no reward for colliding with it.

The environment includes rectangular dark pink bars that divide the space, creating vaguely defined rooms. The bars are heavier and move much slower, providing a slowly changing room layout. Collision with these bars has zero reward for the agent. It is possible that the agent may eventually learn to intentionally collide with bars to change direction without effort.

The default task in the environment is to find and collect food while avoiding obstacles.

## Customization

The PodWorld is highly customizable. For instance, you can do all of the following simply by passing parameters in environment constructor:

* Change the initial impulse so that the environment is slower or faster, changing the difficulty level.
* Change camera FOV, number pixels and how far can it see, impacting the partial observability.
* Change the number of actuators, their placement and strength.
* Change the number of objects, their initial impulses and properties such mass.

See [constructor parameters](podworld/envs/podworld_env.py) for more details.

### Hackability

By default, PodWorld has a simple discrete action space and 1D RGB image (represented by numpy array) as observation space. This allows plug-and-play of several standard algorithms. However, both the action space and the observation space are easily modifiable, for example, to have a continuous action space represented by a 2D vector with magnitude and direction, or to use a 1D depth image instead of an RGB image. It is also extremely easy to change the reward function. The code for all of these is conveniently located in a [single file](podworld/envs/podworld_env.py).

### Potential Applications

PodWorld is designed to be simple and fast, going beyond fixed grid worlds to enable experimentation in areas such as SafeRL and curriculum learning. It presents simple tasks for navigation and obstacle avoidance in a physics + vision-based world, enabling the potential transfer of models to mobile robots in the real world.
