# OnShape to RL (WIP)
This is a work-in-progress repository set to demonstrate the end-to-end workflow from:
- Creating a OnShape CAD drawing
- Export the CAD project to URDF
- Construct a PyBullet Simulation environment using the URDF
- Train the RL agent for the environment

In addition, the repo also contains navie implementation of the following RL algorithms:
-  Proximal Policy Optimization **([PPO](https://github.com/neil-tan/CAD2RL/blob/main/cartpolev1_ppo.py))**
-  Deep Q-Network **([DQN](https://github.com/neil-tan/CAD2RL/blob/main/cartpolev1_dqn_off_policy.py))**
-  Vanilla Policy Gradient **([VPG](https://github.com/neil-tan/CAD2RL/blob/main/cartpolev1_vpg.py))**
-  [Q-Learning](https://github.com/neil-tan/CAD2RL/blob/main/cartpolev1_qtable.py)

## Requirements
- Python 3.9.6+
- PyEnv (optional, but usingful for manage various Python versions on your system)
- VSCode with Python extension (optional, but highly recommended)

## Install
Clone the repo
```bash
$ git clone git@github.com:neil-tan/CAD2RL.git
```

With the correct Python version selected, use these instructions to create a Python virtual environment
```bash
$ cd CAD2RL
$ mkdir -p .pyvenv
$ python -m venv ./.pyvenv/RL
```

Activate the virtual environment
```bash
$ source ./.pyvenv/RL/bin/activate
```

Install the Python packages
```bash
(RL)$ pip install -r requirements.txt
```

## Start Training
Open VSCode and select the Python interpreter from the virtual environment created above. Then, open the Python file you want to run and press `F5` to start training.
