# Welcome to PodWorld

PodWorld is [OpenAI Gym](https://gym.openai.com/) environment for [reinforcement learning](http://incompleteideas.net/book/the-book-2nd.html) experimentations. PodWorld is specifically designed to be partially observable and dynamic (hence abbreviation P.O.D.). We emphasize these two attributes to force agents learn spatial as well as temporal representation of constantly changing environment. In addition, all entities in PodWorld obeys laws of physics allowing for long tail of emergent behaviours that may not be available in games designed with hand crafted rules for human entertainment. 

PodWorld is fast (>500 FPS on usual laptops) without needing GPU, cross-platform and can run in headless mode as well as with render. PodWorld is highly customizable and meant to be hackable so new task definitions can easily be accomodated and difficulty level can easily be changed across several dimensions.

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

PodWord implements full OpenAI Gym interface so you can easily instantiate and use environment. For example, below implements random agent:

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

All objects in PodWord obey laws of physics, have mass and can move due to application of force.

In the screenshot, the agent is colored in red. The agent perceives the world through its 64 pixel 360-degree camera generating 1x64 RGB image of its view. The camera can see only upto N units of distance giving rise to inherent partial observability. The top strip in screenshot shows the agent view. The black color indicates empty space.

 Agent can move in the world using its 16 actuators symmetrically placed around it. The action space simply specifies the 1-based integer index of actuator that agent wants to activate. When an actuator is activated, it generates an impulse of constant magnitude causing motion in opposite direction. To activate no actuators select the the action = 0. 

The environment has obstacles colored in purple that agent should avoid colliding with. If agent does collide with an obstacle, it receives a negative reward proportional to momentum of the obstacle. Small obstacles moving fast or the large obstacle moving slow hurts equally bad.

The environment has food colored in green. Agent should strive to collide with food as much as it can. When agent collides with food, agent receives positive reward proportional to mass of the food object. After collision, food object turns to grey indicating that it has been eaten. After some stochastically distributed time, the food will become green again. When food object is grey, agent receives no reward for colliding with it.

The environment also has rectangular dark pink bars that divides the space, sort of creating seperate rooms. The bars are more heavier so they move much more slowly, sort of creating slowly changing room layout. Collision with these bars has zero reward for the agent. It is possible that agent may eventually learn to intentionally collide with bar to quickly go to certain location.

The default task in the environment is to find and collect food while avoiding obstacles.

## Customization and Hackability

The PodWorld is highly customizable. For instance, you can do all of the following simply by passing parameters in environment constructor:

* Change the initial impulse so that environment is more slow or fast changing the difficulty level.
* Change camera FOV, number pixels and how far can it see impacting the partial observability.
* Change number of actuators, their placement and strength.
* Change number of objects, their initial impulses and properties such mass.

See [constructor parameters](podworld/envs/podworld_env.py) for more details.

### Creating Continuous Action Space

By default, PodWorld has simple discrete action space and 1D RGB image represented by numpy array as observation space. This allows to plug and play with several standard algorithms however it may be more ideal to have agent action space that is continuous 2D vector. This allows agent to control magnitude as well as direction. Similarly one can create observation space that is simply 1D depth image instead of RGB image. It is also extremely easy to change reward function. The code for all of these is conveniently located in [single file](podworld/envs/podworld_env.py).

### Potential Applications

PodWorld is designed to be simple and fast environment that goes beyond fixed grid worlds to enable experimentation in areas such as SafeRL and curriculum learning. It presents simple tasks for navigation and obstacle avoidance in physics + vision based world to enable potential applications in robotics as well.
