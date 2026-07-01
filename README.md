# AutoSweepNav

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-RL-orange)
![Gym](https://img.shields.io/badge/OpenAI%20Gym-compatible-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20prototype-yellow)

Autonomous sweeping and coverage navigation in dynamic human environments.

AutoSweepNav is a research extension built upon Shuijing Liu and collaborators' CrowdNav++ implementation, "Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph" (ICRA 2023). The original project provides attention-based crowd navigation with human trajectory prediction. This repository preserves that foundation while adding autonomous sweeping and coverage-oriented navigation experiments, including PID and ORCA controller support.

This repository does not replace CrowdNav++ and does not claim ownership of the original work. It extends the original codebase for coverage navigation experiments in empty arenas and dynamic human environments.

<p align="center">
  <img src="figures/open.png" width="460" alt="Crowd navigation simulation" />
</p>

## Highlights

| Area | Capabilities |
| --- | --- |
| Sweeping and coverage | Autonomous area sweeping, empty arena coverage, configurable lane-based sweep goals, multi-goal sweep progression |
| Controllers | PID controller support, ORCA-based sweeping support, classical ORCA and Social Force baselines |
| Human environments | Dynamic pedestrians, variable human populations, randomized human attributes and goal changes |
| Prediction | GST-based human trajectory prediction integration, constant-velocity prediction, ground-truth prediction, and no-prediction modes |
| Research workflow | Modular Gym-style environments, PPO/SRNN crowd navigation codepath, saved experiment configs, visualization and evaluation tools |

## Current Status

Autonomous sweeping is functional. PID-based sweeping is implemented. ORCA sweeping is supported. PPO/SRNN sweeping research is still under development. Additional improvements and benchmarking are currently in progress.

Improvements are actively under development, and additional navigation strategies and benchmarking results will be released in future updates.

## Relationship to CrowdNav++

This repository started from Shuijing's CrowdNav++ project:

- Original project: [CrowdNav_Prediction_AttnGraph](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph)
- Paper: [Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph](https://sites.google.com/view/intention-aware-crowdnav/home)
- Preprint: [arXiv:2203.01821](https://arxiv.org/abs/2203.01821)
- Demonstration: [YouTube](https://www.youtube.com/watch?v=nxpxhF019VA)

CrowdNav++ focuses on safe, socially aware robot navigation through dense crowds using attention-based interaction graphs, PPO, SRNN-style policy networks, and human trajectory prediction through GST.

Major additions and adaptations in this repository include:

- Autonomous area sweeping behavior for coverage-style navigation.
- Empty arena sweeping experiments for controlled coverage evaluation.
- Sweep goal generation with configurable axis, lane spacing, margins, and step size.
- PID controller support for direct waypoint tracking.
- ORCA-based sweeping support for collision-aware sweeping in human environments.
- Multi-goal sweep progression that updates robot goals as the robot reaches sweep waypoints.
- Support for evaluating sweeping in both empty and populated arenas.
- Configuration hooks for variable human populations and dynamic human behavior.
- Continued integration with human trajectory prediction for crowd navigation experiments.

The original CrowdNav++ SRNN/PPO infrastructure remains available for crowd navigation research. Sweeping experiments in the current repository should use PID or ORCA rather than SRNN.

## Sweeping Experiments

Current sweeping experiments are designed to use:

- `pid`: PID waypoint tracking for sweep goals.
- `orca`: ORCA-based motion selection for sweeping in dynamic scenes.

Do not use the SRNN policy for sweeping experiments. SRNN and `selfAttn_merge_srnn` remain available for crowd navigation and policy-learning experiments inherited from CrowdNav++, but sweeping is currently evaluated using PID and ORCA.

Key sweeping configuration lives in `crowd_nav/configs/config.py`:

```python
robot.policy = 'orca'      # use 'orca' or 'pid' for current sweeping experiments
robot.sweep = True
robot.sweep_step = 2
robot.sweep_axes = 0       # 0: x-axis sweep, 1: y-axis sweep, 'random': choose per episode
robot.sweep_tail = 1       # show swept area during rendering
robot.sweep_margin = robot.radius + 0.2
robot.sweep_lane_step = robot.radius * 2

sim.empty_arena = False    # True: no humans, False: humans, 'random': mix both
sim.human_num = 10
sim.human_num_range = 0
```

For PID sweeping, configure:

```python
robot.policy = 'pid'
robot.ki = 0.0
robot.kp = 1.2
robot.kd = 0.2
```

## Architecture Overview

```text
                  +-------------------------------+
                  | crowd_nav/configs/config.py   |
                  | experiment and controller cfg |
                  +---------------+---------------+
                                  |
                                  v
  +-------------------+   +-----------------------+   +------------------+
  | crowd_nav/policy  |   | crowd_sim/envs        |   | gst_updated      |
  | PID, ORCA, SRNN   +-->| Gym simulation        |<--+ trajectory pred  |
  +-------------------+   | sweep goal updates    |   +------------------+
                          | dynamic humans        |
                          +-----------+-----------+
                                      |
                                      v
                          +-----------------------+
                          | rl/                   |
                          | PPO, networks, eval   |
                          +-----------+-----------+
                                      |
                                      v
                          +-----------------------+
                          | trained_models/       |
                          | checkpoints, configs  |
                          +-----------------------+
```

## Repository Structure

| Path | Purpose |
| --- | --- |
| `crowd_sim/` | OpenAI Gym-style simulation environments, robot and human dynamics, rendering, sweep goal updates, variable human population environments, and prediction-aware environments. |
| `crowd_nav/` | Navigation policies and project configuration. Includes ORCA, Social Force, PID, SRNN policy wrappers, and `crowd_nav/configs/config.py`. |
| `rl/` | Reinforcement learning stack inherited from CrowdNav++, including PPO, rollout storage, policy networks, vectorized environments, normalization, and evaluation. |
| `trained_models/` | Saved checkpoints, experiment configs, logs, and evaluation outputs for pretrained or local experiments. Includes crowd navigation and sweeping experiment folders. |
| `configs/` | Experiment configuration snapshots are saved under model folders such as `trained_models/<experiment>/configs/`. The live source configuration is `crowd_nav/configs/`. |
| `gst_updated/` | Gumbel Social Transformer trajectory prediction code and pretrained prediction assets used by prediction-aware environments. |
| `figures/` | Images and animations used in documentation and visualization examples. |
| `train.py` | PPO training entry point for learned policy experiments. |
| `test.py` | Evaluation and visualization entry point for PID, ORCA, Social Force, and learned policies. |
| `arguments.py` | Training and evaluation command-line defaults, including output directory, environment name, PPO settings, and network dimensions. |

## Installation

### Requirements

- Linux is recommended.
- Python 3.8 or newer is recommended.
- CUDA-capable GPU is optional for learned policy training and trajectory prediction experiments.
- ORCA experiments may require Python-RVO2.

### 1. Clone

```bash
git clone <this-repository-url>
cd AutoSweepNav
```

If you are working from the original directory name, use that directory instead:

```bash
cd CrowdNav_Prediction_AttnGraph
```

### 2. Create the Environment

```bash
conda env create -f environment.yml
conda activate crowdnav
```

Then install the Python package requirements:

```bash
pip install -r requirements.txt
```

### 3. Optional ORCA Dependency

For ORCA-based policies, install Python-RVO2 if it is not already available:

```bash
git clone https://github.com/sybrenstuvel/Python-RVO2.git
cd Python-RVO2
pip install -e .
cd ..
```

### 4. Verify

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Quick Start

### Evaluate ORCA Sweeping

Set the live configuration in `crowd_nav/configs/config.py`:

```python
robot.policy = 'orca'
robot.sweep = True
sim.empty_arena = False      # or True for empty-arena coverage
```

Run:

```bash
python test.py --model_dir trained_models/sweep_empty_arena_x-axis --test_model 05000.pt --visualize
```

For ORCA and PID, `test.py` does not load a neural policy when `robot.policy` is one of `orca`, `social_force`, or `pid`; the checkpoint argument is kept for compatibility with the existing evaluation interface.

### Evaluate PID Sweeping

Set:

```python
robot.policy = 'pid'
robot.sweep = True
robot.kp = 1.2
robot.ki = 0.0
robot.kd = 0.2
```

Run:

```bash
python test.py --model_dir trained_models/sweep_empty_arena_x-axis --visualize
```

### Train a Learned Crowd Navigation Policy

SRNN/PPO training remains available for crowd navigation experiments inherited from CrowdNav++:

```bash
python train.py
```

Training outputs are written to the directory configured by `--output_dir` in `arguments.py` or passed on the command line:

```bash
python train.py --output_dir trained_models/my_experiment
```

Use SRNN-based policies for crowd navigation research, not for the current sweeping experiments.

## Configuration Guide

### Sweeping and Robot Control

| Parameter | Description |
| --- | --- |
| `robot.policy` | Robot controller or learned policy. Use `pid` or `orca` for current sweeping experiments. |
| `robot.sweep` | Enables autonomous sweep goal progression. |
| `robot.sweep_step` | Goal advancement distance along the active sweep direction. |
| `robot.sweep_axes` | Sweep orientation: `0` for x-axis, `1` for y-axis, or `'random'`. |
| `robot.sweep_margin` | Boundary margin used to keep the robot inside the arena. |
| `robot.sweep_lane_step` | Spacing between adjacent sweep lanes. |
| `robot.sweep_tail` | Enables rendered sweep trace. |
| `robot.kp`, `robot.ki`, `robot.kd` | PID gains used when `robot.policy = 'pid'`. |

### Arena and Humans

| Parameter | Description |
| --- | --- |
| `sim.arena_width`, `sim.arena_height` | Arena dimensions used by the simulator and sweep generator. |
| `sim.empty_arena` | `True` for empty coverage, `False` for human environments, `'random'` for mixed evaluation. |
| `sim.human_num` | Nominal number of humans. Keep this at least `1`; use `sim.empty_arena = True` for empty arenas. |
| `sim.human_num_range` | Variation around the nominal number of humans in variable-population environments. |
| `env.randomize_attributes` | Enables randomized human radius and preferred speed. |
| `humans.random_goal_changing` | Allows human goals to change before reaching the current goal. |
| `humans.end_goal_changing` | Allows new human goals after humans reach their current goals. |

### Prediction Modes

| Mode | Description |
| --- | --- |
| `inferred` | Uses the GST trajectory prediction model in `gst_updated/`. |
| `const_vel` | Uses constant-velocity prediction. |
| `truth` | Uses ground-truth future trajectories where available. |
| `none` | Disables trajectory prediction. |

When `sim.predict_method = 'inferred'`, `env.use_wrapper` must be enabled. The current configuration handles this automatically.

## Evaluation

Use `test.py` for evaluation and visualization:

```bash
python test.py \
  --model_dir trained_models/sweep_empty_arena_x-axis \
  --test_model 05000.pt \
  --visualize
```

Useful options:

| Option | Description |
| --- | --- |
| `--model_dir` | Experiment directory containing configs, checkpoints, and evaluation logs. |
| `--test_model` | Checkpoint filename for learned policies. Kept for interface compatibility with PID/ORCA. |
| `--test_case` | `-1` evaluates the configured test set; nonnegative values repeat a specific case. |
| `--visualize` | Opens a Matplotlib visualization. |
| `--render_traj` | Saves trajectory information when enabled. |
| `--save_slides` | Saves rendered frames for slideshow-style inspection. |

Evaluation logs are written under the selected `trained_models/<experiment>/test/` directory.

## Pretrained and Saved Experiments

The `trained_models/` directory contains saved experiments from the inherited CrowdNav++ workflow and local sweeping experiments.

| Directory | Description |
| --- | --- |
| `trained_models/GST_predictor_rand/` | Crowd navigation policy with GST prediction and randomized human behavior. |
| `trained_models/GST_predictor_non_rand/` | Crowd navigation policy with GST prediction without randomized human behavior. |
| `trained_models/ORCA_no_rand/` | ORCA baseline experiment. |
| `trained_models/SF_no_rand/` | Social Force baseline experiment. |
| `trained_models/sweep_empty_arena/` | Saved sweeping experiment in an empty arena. |
| `trained_models/sweep_empty_arena_x-axis/` | Saved x-axis sweeping experiment. |

Check the `configs/` subdirectory inside each experiment before comparing results, because the saved configuration defines the environment and policy used for that run.

## Roadmap

- Expand benchmarking for empty and human-populated sweeping scenarios.
- Improve coverage metrics and reporting.
- Continue PPO/SRNN-based sweeping research.
- Add more motion planning and coverage planning baselines.
- Improve reproducibility scripts for sweeping experiments.
- Document recommended parameter sets for PID and ORCA sweeping.
- Add richer visualizations for sweep coverage and missed areas.

## Contributing

Pull requests are welcome, especially for:

- Coverage planning.
- Motion planning.
- Navigation algorithms.
- Human-aware robotics.
- Reinforcement learning.
- Robot control.
- Benchmarking and reproducibility.

Please keep contributions scoped, document new configuration options, and include evaluation notes or tests where practical. If your change modifies inherited CrowdNav++ behavior, describe whether it affects crowd navigation, sweeping, or both.

## Citation

If you use the original CrowdNav++ methods, models, or code inherited in this repository, please cite the original work:

```bibtex
@inproceedings{liu2022intention,
  title={Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph},
  author={Liu, Shuijing and Chang, Peixin and Huang, Zhe and Chakraborty, Neeloy and Hong, Kaiwen and Liang, Weihang and Livingston McPherson, D. and Geng, Junyi and Driggs-Campbell, Katherine},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023},
  pages={12015--12021}
}
```

The SRNN crowd navigation foundation is also associated with:

```bibtex
@inproceedings{liu2020decentralized,
  title={Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning},
  author={Liu, Shuijing and Chang, Peixin and Liang, Weihang and Chakraborty, Neeloy and Driggs-Campbell, Katherine},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021},
  pages={3517--3524}
}
```

If you use this extension for autonomous sweeping or coverage navigation experiments, please cite this repository as an extension:

```bibtex
@misc{autosweepnav,
  title = {AutoSweepNav: Autonomous Sweeping and Coverage Navigation in Dynamic Human Environments},
  howpublished = {\url{<repository-url>}},
  note = {Extension of Shuijing Liu et al.'s CrowdNav++ implementation for autonomous sweeping and coverage navigation},
  year = {2026}
}
```

Replace `<repository-url>` with the public URL of this repository when citing.

## Acknowledgements

This repository builds on the work of Shuijing Liu and collaborators, whose CrowdNav++ implementation and related papers provide the foundation for the crowd navigation and trajectory prediction workflow used here.

Thanks to:

- Shuijing and collaborators for the original CrowdNav++ project.
- The CrowdNav++ research codebase and papers.
- The GST authors for the trajectory prediction model integrated into the original workflow.
- The open-source robotics community for tools, baselines, and shared research infrastructure.

## License

This repository follows the included `LICENSE`. The inherited CrowdNav++ components remain credited to their original authors.
