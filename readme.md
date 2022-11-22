**Status:** Maintenance (expect bug fixes and minor updates)

spinningup-pytorch
==================================

This is a clone from [Spinning Up](https://github.com/openai/spinningup) with the goal for using the latest pytorch version. Tensorflow-based codes were removed from the original repository. For beginners, please get started at [spinningup.openai.com](https://spinningup.openai.com)!

### TDOD List

add different activation functions for pi and v:

- [ ] VPG
- [ ] TRPO
- [x] PPO
- [ ] DDPG
- [x] TD3

### Installation

```
git clone git@github.com:haha1227/spinningup-pytorch.git
cd spinningup-pytorch
pip install -e .
```

To uninstall it, run the following:
```
pip uninstall spinup
```
### Usage

##### Launching from the Command Line:
```
python -m spinup.run [algo name] [experiment flags]
```
*E.g.*:
```
python3 -m spinup.run ppo --env CartPole-v1 --exp_name walker --hid [64,64] --pi_act torch.nn.Tanh
```
or run the same algorithm with many possible hyperparameters:
```
python3 -m spinup.run ppo --env CartPole-v1 --exp_name walker --hid [32] [64,64] --pi_act torch.nn.Tanh torch.nn.ReLU
```
##### Launching from Scripts:
```
from spinup import ppo_pytorch as ppo
import gym
import torch

seed = 0
exp_name = 'walker'
env_fn = lambda : gym.make('CartPole-v1')
ac_kwargs = dict(hidden_sizes=[64,64], pi_output_activation=torch.nn.Tanh)
logger_kwargs = dict(output_dir=''.join([exp_name, '_s', str(seed)]), exp_name=exp_name)

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, seed = seed, logger_kwargs=logger_kwargs)
```
##### Launching from Scripts for multiple hyperparameters:
```
from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()

    eg = ExperimentGrid(name='ppo-pyt-bench')
    eg.add('env_name', 'CartPole-v1', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 10)
    eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
    eg.run(ppo_pytorch, num_cpu=args.cpu)
```
Note: the default trained model is saved in the [data](./data/) folder.

### Inference
```
python -m spinup.run test_policy path/to/output_directory
```
or using scripts:
```
from spinup.utils.test_policy import load_policy_and_env, run_policy
import your_env
_, get_action = load_policy_and_env('/path/to/output_directory')
env = your_env.make()
run_policy(env, get_action)
```
### Plotting Results
```
python -m spinup.run plot [path/to/output_directory ...] [--legend [LEGEND ...]]
    [--xaxis XAXIS] [--value [VALUE ...]] [--count] [--smooth S]
    [--select [SEL ...]] [--exclude [EXC ...]]
```
Please check the [details](https://spinningup.openai.com/en/latest/user/plotting.html).

### Citing Spinning Up

If you reference or use Spinning Up in your research, please cite:

```
@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}
```
