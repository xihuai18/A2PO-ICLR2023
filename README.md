Codebase for [Order Matters: Agent-by-agent Policy Optimization](https://openreview.net/forum?id=Q-neeWNVv1). **Please note that this repo is currently undergoing reconstruction and may potentially contain bugs. As a result, its performance may not be consistent with that which was reported in the paper.**

## 1. Usage

CTDE MARL algorithms for StarCraft II Multi-agent Challenge (SMAC), PettingZoo, Multi-agent Particle Environment, Multi-agent MUJOCO, and Google Research Football (comming later).

This repo is heavily based on https://github.com/marlbenchmark/on-policy.

Implemented algorithms include:
- [x] MAPPO
- [x] CoPPO (without advantage-mix / credit assignment) 
- [x] HAPPO
- [x] A2PO

Parameter sharing is recommended for SMAC, while separated parameters are recommended for other environments.

## 2. Results

Monotonic Bounds

<img src="./media/bound.png" height = "300" alt="bounds" align=center />
<img src="./media/vis_bound.png" height = "300" alt="bounds" align=center />

Cooperative Break Through

![](./media/cooperative-break-through.gif)

Serve, Pass and Shoot

![](./media/serve-pass-and-shoot.gif)

## 3. Installation

``` Bash
# create conda environment
conda create -n co-marl python==3.8
conda activate co-marl
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
pip install psycopg2-binary setproctitle absl-py pysc2 gym tensorboardX
# sudo apt install build-essential -y
pip install Cython
pip install sacred aim
# install on-policy package
pip install -e .
```

or 

```shell
bash install.sh
```


### 3.1 Install StarCraftII

```shell
bash install_sc2.sh
```

### 3.2 Install PettingZoo

```shell
bash install_pettingzoo.sh
```

### 3.3 Install Multi-agent Mujoco

```shell
bash install_mujoco.sh
```

### 3.4 Install Google Search Football
```shell
bash install_grf.sh
```

## 4.Train
Here we use training MMM2 (a hard task in SMAC) as an example:
```
sh run_scripts/SMAC/MMM2.sh
```