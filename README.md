# AutoSweepCrowdNav

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-orange)
![Gym](https://img.shields.io/badge/OpenAI%20Gym-compatible-lightgrey)
![Status](https://img.shields.io/badge/status-research%20extension-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

Autonomous sweeping and coverage-oriented crowd navigation experiments built on top of Shuijing Liu and collaborators' CrowdNav++ implementation.

This repository started from [Shuijing725/CrowdNav_Prediction_AttnGraph](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph), the implementation of **"Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph"** from ICRA 2023. The original project remains the foundation for the crowd navigation, PPO/SRNN, and GST trajectory prediction components. This fork extends that framework with sweeping, rectangular simulation geometry, ellipse based crowd generation, PID control, ORCA-based sweeping, empty-arena experiments, and coverage-oriented evaluation hooks.

This project does not replace CrowdNav++ and does not claim ownership of the original work. It is a research extension for autonomous sweeping in empty and human-populated environments.

<p align="center">
  <img src="figures/open.png" width="460" alt="Crowd navigation simulation" />
</p>

## Relationship to CrowdNav++

CrowdNav++ provides the original simulation, attention-based policy architecture, PPO training pipeline, human trajectory prediction integration, and social navigation evaluation workflow.

This repository keeps that foundation and adds experimental sweeping functionality:

| Area | Original CrowdNav++ | This Extension |
| --- | --- | --- |
| Main task | Crowd navigation to a single goal | Multi-goal sweeping and coverage-oriented navigation |
| Arena model | square-style assumptions | Rectangular arena dimensions and ellipse-based human generation |
| Robot policy default | `selfAttn_merge_srnn` | `orca` for current sweeping experiments |
| Controllers | ORCA, Social Force, SRNN policies | Adds PID and uses ORCA/PID for sweeping |
| Human generation | Circle crossing | Ellipse crossing, controlled entry/exit hooks, optional boundary starts, experimental groups |
| Empty arena | Not a primary experiment mode | Empty arena mode with dummy human compatibility |
| Rendering | Crowd navigation visualization | Sweep trace, rectangular bounds, group labels |
| Evaluation | Aggregate success/collision/timeout | Adds empty-vs-crowded episode reporting |

SRNN/PPO remains available for inherited crowd navigation experiments. Current sweeping experiments are designed around **PID** and **ORCA**, not SRNN.

## Current Status

Autonomous sweeping is functional. PID-based sweeping is implemented. ORCA sweeping is supported. Empty-arena logic, rectangular arena support, ellipse crowd generation, controlled entry/exit hooks, and group spawning are present in the codebase.

PPO/SRNN sweeping research is still under development. Coverage metrics, coverage-specific rewards, human group behavior improvements, controlled entry/exit refinement, and broader benchmarking are in progress.

Improvements are actively under development, and additional navigation strategies and benchmarking results will be released in future updates.

## Implemented Features

| Feature | Status | Notes |
| --- | --- | --- |
| Autonomous area sweeping | Implemented | Sweep waypoints are generated and updated during an episode. |
| Empty arena mode | Implemented | Uses a dummy human slot to preserve observation tensor compatibility. |
| Rectangular arena | Implemented | Adds `arena_width` and `arena_height`; active paths use rectangular bounds. |
| Ellipse crowd generation | Implemented | Replaces circle crossing with ellipse crossing in active environment paths. |
| Multi-goal sweeping | Implemented | Reaching intermediate goals updates the next sweep goal instead of ending immediately. |
| Sweep stop condition | Implemented | Episodes end when the computed final sweep waypoint is reached. |
| PID controller | Implemented | Added as a non-trainable policy in `crowd_nav/policy/pid.py`. |
| ORCA sweeping | Implemented | ORCA follows generated sweep goals through the existing policy interface. |
| Variable human populations | Inherited and modified | Add/remove logic updated for max-human and new generation behavior. |
| Controlled entry/exit hooks | Experimental | Boundary/gate-style spawning and dynamic population updates support controlled crowd flow experiments. |
| Human group behavior | In progress | Group members can be generated and labeled; coordinated group behavior is unfinished. |
| Sweep visualization | Implemented with caveat | Sweep tail is drawn; inspect render debug code before relying on batch rendering. |
| Empty/crowded evaluation split | Implemented | Evaluation reports separate counts and rates for empty and crowded episodes. |

## Sweeping Experiments

Current sweeping experiments should use:

- `pid`: PID waypoint tracking for generated sweep goals.
- `orca`: ORCA-based collision-aware motion toward generated sweep goals.

Do **not** use SRNN for current sweeping experiments. SRNN and `selfAttn_merge_srnn` remain available for crowd navigation and future RL-based sweeping research.

Key configuration lives in `crowd_nav/configs/config.py`:

```python
robot.policy = 'orca'      # use 'orca' or 'pid' for current sweeping experiments
robot.sweep = True
robot.sweep_step = 2
robot.sweep_axes = 0       # 0: x-axis, 1: y-axis, 'random': choose per episode
robot.sweep_tail = 1
robot.sweep_margin = robot.radius + 0.2
robot.sweep_lane_step = robot.radius * 2

sim.empty_arena = False    # True, False, or 'random'
sim.arena_width = 10
sim.arena_height = 8
```

For PID:

```python
robot.policy = 'pid'
robot.kp = 1.2
robot.ki = 0.0
robot.kd = 0.2
```

## Architecture Overview

```text
crowd_nav/configs/config.py
        |
        v
crowd_nav/policy/
  ORCA, Social Force, SRNN, PID
        |
        v
crowd_sim/envs/
  rectangular arena, ellipse humans, empty arena,
  sweep waypoint updates, dynamic populations
        |
        +--------------------+
        |                    |
        v                    v
rl/ PPO, evaluation      gst_updated/ GST prediction
        |
        v
trained_models/
```

## Repository Structure

| Path | Description |
| --- | --- |
| `crowd_sim/` | Gym-style simulation environments, robot/human dynamics, ellipse generation, empty arena handling, sweep logic, rendering, and prediction-aware environments. |
| `crowd_nav/` | Navigation policies and configuration. Includes ORCA, Social Force, SRNN wrappers, the added PID controller, and experiment settings. |
| `rl/` | PPO, rollout storage, network definitions, vectorized environments, normalization, and evaluation utilities inherited from CrowdNav++. |
| `trained_models/` | Saved checkpoints, copied configs, logs, and evaluation outputs for inherited and local experiments. |
| `gst_updated/` | Gumbel Social Transformer trajectory prediction code and pretrained prediction assets inherited from the original workflow. |
| `figures/` | Documentation and visualization assets. |
| `train.py` | PPO training entry point for learned policy experiments. |
| `test.py` | Evaluation and visualization entry point for PID, ORCA, Social Force, and learned policies. |
| `arguments.py` | Training/evaluation command-line defaults. |
| `environment.yml` | Conda environment specification added in this extension. |

## Installation

### Requirements

- Linux is recommended.
- Python 3.8+ is recommended; `environment.yml` currently pins Python 3.10.
- CUDA is optional but useful for learned policies and prediction workflows.
- ORCA experiments may require Python-RVO2.

### Conda Setup

```bash
conda env create -f environment.yml
conda activate crowdnav
pip install -r requirements.txt
```

### Optional ORCA Dependency

```bash
git clone https://github.com/sybrenstuvel/Python-RVO2.git
cd Python-RVO2
pip install -e .
cd ..
```

### Verify

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Quick Start

### ORCA Sweeping

Set:

```python
robot.policy = 'orca'
robot.sweep = True
```

Run:

```bash
python test.py --model_dir trained_models/sweep_empty_arena_x-axis --test_model 05000.pt --visualize
```

For ORCA and PID, the neural checkpoint is not loaded; the checkpoint argument is retained for compatibility with the original evaluation interface.

### PID Sweeping

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

### Empty Arena Sweeping

Set:

```python
sim.empty_arena = True
robot.sweep = True
robot.policy = 'pid'  # or 'orca'
```

The simulator keeps a dummy human entry so downstream observation and recurrent-policy utilities continue to receive valid tensor shapes.

### Mixed Empty/Crowded Evaluation

Set:

```python
sim.empty_arena = 'random'
```

Evaluation logs report empty and crowded episode counts and success/collision/timeout rates separately.

### Learned Crowd Navigation

Inherited PPO/SRNN training remains available for crowd navigation:

```bash
python train.py --output_dir trained_models/my_experiment
```

Use learned SRNN-style policies for crowd navigation research unless you are explicitly developing the unfinished RL sweeping path.

## Configuration Reference

### New Environment Settings

| Variable | Purpose |
| --- | --- |
| `env.fig_size_x`, `env.fig_size_y` | Matplotlib figure size for visualization. |
| `env.axes_equal` | Whether rendered axes use equal aspect ratio. |
| `env.axes_visible` | Whether rendered axes are visible. |
| `reward.step_penalty` | Configured step penalty; currently loaded but not fully integrated into the main sweep reward path. |

### New Simulation Settings

| Variable | Purpose |
| --- | --- |
| `sim.arena_width`, `sim.arena_height` | Rectangular arena dimensions. |
| `sim.ellipse_a`, `sim.ellipse_b` | Ellipse radii used for human crossing generation. |
| `sim.start_at_boundary` | Optional boundary/gate-style human spawning for controlled entry/exit experiments. |
| `sim.empty_arena` | `True`, `False`, or `'random'` for empty/crowded experiments. |
| `sim.group` | Enables in-progress human group behavior experiments. |
| `sim.group_size` | Number of nearby humans to spawn in a group. |

### New Robot Settings

| Variable | Purpose |
| --- | --- |
| `robot.sweep` | Enables sweep waypoint generation. |
| `robot.sweep_step` | Distance advanced along the active sweep direction. |
| `robot.sweep_axes` | Sweep orientation: `0`, `1`, or `'random'`. |
| `robot.sweep_tail` | Enables swept-area trace rendering. |
| `robot.sweep_margin` | Keeps sweep goals away from arena boundaries. |
| `robot.sweep_lane_step` | Spacing between sweep lanes. |
| `robot.kp`, `robot.ki`, `robot.kd` | PID gains used by `robot.policy = 'pid'`. |

## Evaluation

```bash
python test.py \
  --model_dir trained_models/sweep_empty_arena_x-axis \
  --test_model 05000.pt \
  --visualize
```

Useful options:

| Option | Description |
| --- | --- |
| `--model_dir` | Experiment directory for logs/checkpoints/config snapshots. |
| `--test_model` | Checkpoint filename for learned policies; unused for PID/ORCA execution. |
| `--test_case` | `-1` evaluates the configured test set; nonnegative values repeat one case. |
| `--visualize` | Opens Matplotlib visualization. |
| `--render_traj` | Saves trajectory information. |
| `--save_slides` | Saves rendered frames. |

Evaluation reports:

- Overall success, collision, and timeout rates.
- Average navigation time.
- Path length.
- Intrusion ratio and minimum-distance statistics.
- Empty arena episode counts and rates.
- Crowded arena episode counts and rates.

## Implementation Notes

The comparison against upstream CrowdNav++ identified several important caveats:

- Coverage is currently represented by generated sweep waypoints and sweep-tail visualization. A complete coverage metric or coverage-specific reward is not yet implemented.
- `reward.step_penalty` exists in config but is not fully applied in the main reward path.
- Human group behavior is in progress: spawning and labels exist, but coordinated group behavior is unfinished.
- Controlled entry/exit is experimental and currently represented through boundary/gate-style spawning and dynamic population changes.
- One rendering path in `crowd_sim/envs/crowd_sim_var_num.py` has carried debug breakpoint code around sweep rendering. Keep that code removed/commented before relying on visualization runs.
- `test.py` currently uses the live config from `crowd_nav/configs/config.py`; saved experiment config loading is commented out.
- `sim.human_num` should remain at least `1`; use `sim.empty_arena = True` for empty arenas.

## Work in Progress

The following features are present, partial, or planned for further development:

- Human group behaviour improvements.
- Controlled entry and exit improvements.
- Empty arena sweeping benchmarks.
- Rectangle arena polishing and validation.
- Ellipse crowd generation validation.
- Coverage navigation metrics and reporting.
- Coverage-specific reward logic.
- Additional controllers beyond PID and ORCA.
- Future reinforcement learning improvements for PPO/SRNN-based sweeping.
- Removal of debug artifacts and cleanup of experimental code paths.
- More complete benchmarking against crowd-navigation and coverage baselines.

## Files Changed Relative to CrowdNav++

Added:

- `crowd_nav/policy/pid.py`
- `environment.yml`

Modified:

- `README.md`
- `arguments.py`
- `crowd_nav/configs/config.py`
- `crowd_nav/policy/policy_factory.py`
- `crowd_sim/envs/crowd_sim.py`
- `crowd_sim/envs/crowd_sim_pred.py`
- `crowd_sim/envs/crowd_sim_pred_real_gst.py`
- `crowd_sim/envs/crowd_sim_var_num.py`
- `crowd_sim/envs/crowd_sim_var_num_collect.py`
- `crowd_sim/envs/utils/agent.py`
- `requirements.txt`
- `rl/evaluation.py`
- `test.py`
- `train.py`
- `trained_models/GST_predictor_rand/arguments.py`

Removed:

- No tracked files were removed relative to the upstream repository.

## Roadmap

- Add explicit coverage metrics and coverage maps.
- Add benchmark scripts for PID, ORCA, and future learned sweeping policies.
- Stabilize group behavior and dynamic population experiments.
- Clean up debug code and saved-config loading behavior.
- Document reproducible experiment presets.
- Extend controller comparisons.
- Investigate PPO/SRNN policies for sweeping after classical-controller baselines are stable.

## Contributing

Pull requests are welcome for:

- Coverage planning.
- Motion planning.
- Navigation algorithms.
- Human-aware robotics.
- Reinforcement learning.
- Robot control.
- Benchmarking and reproducibility.

Please keep changes scoped and document whether they affect inherited CrowdNav++ behavior, sweeping extensions, or both.

## Citation

If you use the original CrowdNav++ implementation or inherited methods, cite:

```bibtex
@inproceedings{liu2022intention,
  title={Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph},
  author={Liu, Shuijing and Chang, Peixin and Huang, Zhe and Chakraborty, Neeloy and Hong, Kaiwen and Liang, Weihang and Livingston McPherson, D. and Geng, Junyi and Driggs-Campbell, Katherine},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023},
  pages={12015--12021}
}
```

The inherited SRNN crowd navigation foundation is associated with:

```bibtex
@inproceedings{liu2020decentralized,
  title={Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning},
  author={Liu, Shuijing and Chang, Peixin and Liang, Weihang and Chakraborty, Neeloy and Driggs-Campbell, Katherine},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021},
  pages={3517--3524}
}
```

If you use this extension for autonomous sweeping or coverage navigation experiments, cite this repository as an extension:

```bibtex
@misc{autosweepnav,
  title = {AutoSweepNav: Autonomous Sweeping and Coverage-Oriented Navigation in Dynamic Human Environments},
  howpublished = {\url{<repository-url>}},
  note = {Research extension of Shuijing Liu et al.'s CrowdNav++ implementation},
  year = {2026}
}
```

Replace `<repository-url>` with the public URL of this repository.

## Acknowledgements

Thanks to:

- Shuijing Liu and collaborators for the original CrowdNav++ project.
- The CrowdNav++ authors for the attention-based crowd navigation framework.
- The GST authors for trajectory prediction components used by the original workflow.
- The open-source robotics community for simulation, learning, and navigation tools.

## License

This repository follows the included `LICENSE`. Inherited CrowdNav++ components remain credited to their original authors.
