# Welcome to PodWorld

PodWorld is [OpenAI Gym](https://gym.openai.com/) environment for [reinforcement learning](http://incompleteideas.net/book/the-book-2nd.html) experimentations. PodWorld is specifically designed to be partially observable and dynamic (hence abbreviation P.O.D.). We emphasize these two attributes to force agents learn spatial as well as temporal representations that must go beyond simple memorization. In addition, all entities in PodWorld must obey laws of physics allowing for long tail of emergent observations that may not appear in games designed with arbitrary hand crafted rules for human entertainment. PodWorld is designed to be highly customizable as well as hackable with fairly simple and minimal code base.

PodWorld is designed to be fast (>500 FPS on usual laptops) without needing GPU and run cross platform in headless mode or with render.

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

In the screenshot, the agent is colored in red. The agent perceives the world through its 64 pixel 360-degree camera generating 1x64 RGB image of its view. The camera can see only upto N units giving rise to inherent partial observability. The top strip in screen shot shows the agent view. The black color indicates empty space.

 Agent can move in the world using its 16 actuators symmetrically placed around it. The action space simply specifies the 1-based index of actuator to activate. To activate no actuators set the action=0. When an actuator is activated, it generates a fixed thrust causing motion in opposite direction.

The environment has obstacles colored in purple that agent should avoid colliding with. If agent does collide with an obstacle, it receives a negative reward proportional to momentum of the obstacle.

The environment has food colored in green. Agent should strive to collide with food as much as it can. When agent collides with food, agent receives positive reward proportional to mass of the food object. After collision, food object turns to grey and will become green again after some amount of time. When food object is grey, agent receives no reward for colliding with it.

The environment also has rectangular dark pink bars that divides the space, sort of creating rooms that must be explored. The bars are assigned higher mass but they still move causing dynamic rooms that slowly change over time. Collision with these bars has zero reward.

The default task in the environment is to find and collect food while avoiding obstacles.

## Customization and Hackability

The PodWorld is highly customizable. For instance, you can do all of the following simply by passing parameters in environment constructor:

* Change the initial impulse causing environment that is slowly changing or very dynamic.
* Change camera FOV and number pixels.
* Change number of actuators, their placement and strength.
* Change number of objects, their initial impulses, mass.

See [constructor parameters](podworld/envs/podworld_env.py) for more details.

### Creating Continuous Action Space

By default, PodWorld has simple discrete action space and 1D RGB image represented by numpy array as observation space. This allows to plug and play with several standard algorithms however it may be more ideal to have agent action space that is continuous 2D vector representing thrust. This allows agent to control magnitude as well as direction. Similarly one can create observation space that is simply 1D depth image instead of RGB image. It is also extremely easy to change reward function. The code all of these is conveniently located in [single file](podworld/envs/podworld_env.py).

### Potential Applications

PodWorld is designed to be simple yet fast environment that goes beyond grid worlds to enable experimentation in areas such as SafeRL and curriculum learning. It present simple tasks for navigation and obstacle avoidance in physics + vision based world to enable potential applications in robotics as well.
