# CrowdNav++: Intention-Aware Crowd Navigation with Attention-Based Interaction Graph

This repository contains the implementation of **"Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph"** published at **ICRA 2023**. The work addresses safe and socially-aware robot navigation in dense, interactive crowds using graph neural networks with attention mechanisms and human trajectory prediction.

**Paper & Resources:**
- 📄 **Paper**: [ICRA 2023 Proceedings](https://sites.google.com/view/intention-aware-crowdnav/home)
- 📝 **Preprint**: [arXiv:2203.01821](https://arxiv.org/abs/2203.01821)
- 🎬 **Demonstrations**: [YouTube Video](https://www.youtube.com/watch?v=nxpxhF019VA)

**Latest Updates:**
- 🔗 Follow-up work: [HEIGHT: Heterogeneous Interaction Graph Transformer](https://github.com/Shuijing725/CrowdNav_HEIGHT)
- 🔄 Sim2Real transfer: [CrowdNav Sim2Real Tutorial](https://github.com/Shuijing725/CrowdNav_Sim2Real_Turtlebot)
- 📚 Curated resources: [Awesome Robot Social Navigation](https://github.com/Shuijing725/awesome-robot-social-navigation)

---

## Overview

### Problem Statement

Robot navigation in crowded environments requires understanding both:
1. **Interactions** between agents (robot and pedestrians)
2. **Intentions** of pedestrians (where they plan to go)

Existing RL-based methods often fail to model these factors explicitly, leading to poor navigation performance and socially inappropriate behaviors.

### Our Solution

We propose a **Graph Neural Network with Attention Mechanisms** that:
- ✅ Captures **heterogeneous interactions** among all agents through space and time
- ✅ Predicts **future pedestrian trajectories** to infer intentions
- ✅ Prevents collision by avoiding predicted pedestrian paths
- ✅ Learns navigation policies using **Model-Free Reinforcement Learning (PPO)**
- ✅ Successfully **transfers to real-world robots** (TurtleBot 2i)

<p align="center">
<img src="figures/open.png" width="450" alt="Robot navigation in dense crowd" />
</p>

### Key Features

- **Attention-Based Interaction Graph**: Captures heterogeneous robot-human and human-human interactions
- **Trajectory Prediction**: Integrates Gumbel Social Transformer (GST) for human motion forecasting
- **Flexible Prediction Methods**: Supports inferred, ground-truth, constant-velocity, and no-prediction modes
- **Randomized Human Behaviors**: Realistic crowd simulation with varying speeds, radii, and goals
- **Empty Arena Mode**: Train and test with no pedestrians for baseline comparison
- **Sweep Functionality**: Test robot navigation with controlled motion patterns
- **Visualization & Analysis**: Built-in visualization and training curve plotting

---

## Repository Structure

```
CrowdNav_Prediction_AttnGraph/
├── arguments.py                      # Training/testing hyperparameter configuration
├── train.py                          # Main training script
├── test.py                           # Evaluation and visualization script
├── plot.py                           # Training curve visualization
├── collect_data.py                   # Data collection utilities
├── environment.yml                   # Conda environment specification
├── requirements.txt                  # Python package dependencies
├── LICENSE                           # MIT License
│
├── crowd_nav/                        # Navigation policies and configurations
│   └── configs/
│       ├── config.py                 # Simulation, reward, and environment settings
│       └── __init__.py
│
├── crowd_sim/                        # Simulation environment (OpenAI Gym compatible)
│   ├── envs/
│   │   ├── crowd_sim.py              # Base environment
│   │   ├── crowd_sim_var_num.py      # Variable number of pedestrians
│   │   ├── crowd_sim_pred.py         # With trajectory prediction
│   │   ├── crowd_sim_pred_real_gst.py # With GST prediction
│   │   ├── crowd_sim_var_num_collect.py
│   │   └── utils/                    # Robot, human, action, state utilities
│   └── ...
│
├── rl/                               # Reinforcement learning components
│   ├── ppo/                          # PPO algorithm implementation
│   ├── networks/
│   │   ├── model.py                  # Policy network architecture
│   │   ├── network_utils.py
│   │   ├── storage.py                # Rollout buffer
│   │   ├── envs.py                   # Environment wrapper
│   │   └── ...
│   ├── evaluation.py                 # Evaluation metrics
│   └── ...
│
├── gst_updated/                      # Gumbel Social Transformer predictor (inference only)
│   ├── results/                      # Pretrained trajectory prediction models
│   ├── run/
│   └── ...
│
├── trained_models/                   # Pretrained navigation policies
│   ├── GST_predictor_no_rand/        # Ours (no human randomization)
│   ├── GST_predictor_rand/           # Ours (with human randomization)
│   ├── ORCA_no_rand/                 # ORCA baseline
│   ├── SF_no_rand/                   # Social Force baseline
│   └── ...
│
└── figures/                          # Images and visualizations
    ├── open.png
    ├── visual.gif
    ├── rewards.png
    └── losses.png
```

---

## Installation

### Requirements

- **Python**: 3.8+ (tested with 3.8 and 3.10)
- **OS**: Linux (Ubuntu 18.04+)
- **GPU**: Optional but recommended (CUDA-capable device)

### Step 1: Clone Repository

```bash
git clone https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph.git
cd CrowdNav_Prediction_AttnGraph
```

### Step 2: Setup Conda Environment

```bash
conda env create -f environment.yml
conda activate crowdnav
```

This installs Python 3.10, PyTorch 1.12.1, and base dependencies.

### Step 3: Install Additional Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy==1.23.5` – Numerical computing
- `pandas==1.5.2` – Data handling
- `gym==0.15.7` – Environment interface
- `tensorflow-gpu==2.11.0` – GPU support (CPU alternative available)
- `matplotlib==3.6.2` – Visualization
- `Cython` – Fast Python extensions

### Step 4: Install OpenAI Baselines (Optional)

Only needed if training from scratch. Provides PPO algorithm utilities:

```bash
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ..
```

### Step 5: Install Python-RVO2 (Optional)

For ORCA-based pedestrian dynamics:

```bash
git clone https://github.com/sybrenstuvel/Python-RVO2.git
cd Python-RVO2
pip install -e .
cd ..
```

### Step 6: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Quick Start

### Training a Navigation Policy

**1. Configure the environment** (`crowd_nav/configs/config.py`):

```python
# Set prediction method
sim.predict_method = 'inferred'  # Options: 'inferred', 'const_vel', 'truth', 'none'

# Enable human randomization for realistic scenarios
env.randomize_attributes = True
humans.random_goal_changing = False

# Choose arena size
sim.human_num = 5
sim.empty_arena = 'random'  # 'random', True (empty), or False (crowded)
```

**2. Configure training** (`arguments.py`):

```python
parser.add_argument('--output_dir', type=str, default='trained_models/my_model')
parser.add_argument('--num-processes', type=int, default=4)  # Parallel environments
parser.add_argument('--num-steps', type=int, default=2048)   # Rollout length
```

**3. Run training**:

```bash
python train.py
```

**Output**: Checkpoints and configs saved to `trained_models/my_model/`

### Testing & Visualization

**1. Configure test parameters** (`test.py`, lines 20-33):

```python
parser.add_argument('--model_dir', type=str, default='trained_models/GST_predictor_rand')
parser.add_argument('--visualize', default=True, action='store_true')
parser.add_argument('--test_case', type=int, default=-1)  # -1 for 500 different cases
parser.add_argument('--test_model', type=str, default='41665.pt')
```

**2. Run evaluation**:

```bash
python test.py
```

**Output**: Results logged to `trained_models/<model_dir>/test/` and displayed on terminal.

**Example visualization**:
<img src="figures/visual.gif" width="420" alt="Navigation visualization in crowd" />

### Plotting Training Curves

```bash
python plot.py
```

Generates plots for:
- Episode rewards over time
- Training losses

<img src="figures/rewards.png" width="370" alt="Training rewards" /> 
<img src="figures/losses.png" width="370" alt="Training losses" />

---

## Configuration Guide

### Simulation Configuration (`crowd_nav/configs/config.py`)

#### Environment Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env.time_limit` | 50 | Max timesteps per episode |
| `env.time_step` | 0.25 | Simulation time step (seconds) |
| `env.randomize_attributes` | True | Randomize human radius and velocity |
| `env.val_size` | 100 | Validation episodes |
| `env.test_size` | 500 | Test episodes |

#### Simulation Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sim.arena_width` | 10 | Arena width (meters) |
| `sim.arena_height` | 8 | Arena height (meters) |
| `sim.human_num` | 5 | Number of pedestrians |
| `sim.human_num_range` | 3 | Variation in pedestrian count |
| `sim.empty_arena` | 'random' | 'random', True (empty), False (crowded) |
| `sim.predict_method` | 'inferred' | Prediction: 'inferred', 'const_vel', 'truth', 'none' |
| `sim.predict_steps` | 5 | Future timesteps to predict |

#### Human Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `humans.policy` | 'orca' | Pedestrian policy: 'orca', 'social_force' |
| `humans.v_pref` | 1.0 | Preferred velocity (m/s) |
| `humans.radius` | 0.3 | Pedestrian radius (meters) |
| `humans.FOV` | 2.0 | Field of view (× π radians) |
| `humans.random_goal_changing` | False | Random goal changes |
| `humans.end_goal_changing` | False | Goal changes upon arrival |

#### Reward Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reward.success_reward` | 10 | Success bonus |
| `reward.collision_penalty` | -20 | Collision penalty |
| `reward.discomfort_dist` | 0.25 | Minimum comfort distance (meters) |
| `reward.discomfort_penalty_factor` | 10 | Discomfort penalty multiplier |
| `reward.gamma` | 0.99 | Discount factor |

### Network Configuration (`arguments.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--env_name` | — | Environment: `CrowdSimPredRealGST-v0`, `CrowdSimPred-v0`, `CrowdSimVarNum-v0` |
| `--use_self_attn` | True | Enable human-human attention |
| `--use_hr_attn` | True | Enable robot-human attention |
| `--num_processes` | 4 | Parallel environment instances |
| `--num_steps` | 2048 | Rollout buffer length |
| `--num_mini_batch` | 32 | PPO mini-batch size |
| `--lr` | 2.5e-4 | Learning rate |
| `--cuda_deterministic` | False | Deterministic CUDA (slower) |
| `--seed` | 425 | Random seed |

---

## Pretrained Models

### Available Models

| Method | Location | Checkpoint | Description |
|--------|----------|------------|-------------|
| **Ours (No Randomization)** | `trained_models/GST_predictor_no_rand` | `41200.pt` | Best performance with static humans |
| **Ours (With Randomization)** | `trained_models/GST_predictor_rand` | `41665.pt` | Robust to dynamic human behaviors |
| **ORCA Baseline** | `trained_models/ORCA_no_rand` | `00000.pt` | Classical collision avoidance |
| **Social Force** | `trained_models/SF_no_rand` | `00000.pt` | Physics-based pedestrian model |

### Download & Test

```bash
# Test our model (with prediction)
python test.py \
  --model_dir trained_models/GST_predictor_rand \
  --test_model 41665.pt \
  --visualize True

# Compare with baseline (ORCA)
python test.py \
  --model_dir trained_models/ORCA_no_rand \
  --test_model 00000.pt
```

---

## Advanced Features

### 1. Empty Arena Mode

Train and test in environments with no pedestrians for baseline comparisons:

```python
# In crowd_nav/configs/config.py
sim.empty_arena = True   # Disable pedestrians
```

### 2. Human Randomization

Add variability to pedestrian behaviors:

```python
# In crowd_nav/configs/config.py
env.randomize_attributes = True      # Random radius and velocity
humans.random_goal_changing = True   # Random goal changes
```

### 3. Trajectory Prediction Methods

Choose how to predict human motion:

```python
# In crowd_nav/configs/config.py
sim.predict_method = 'inferred'  # Learned GST model
sim.predict_method = 'const_vel' # Constant velocity
sim.predict_method = 'truth'     # Ground truth (oracle)
sim.predict_method = 'none'      # No prediction
```

**Note**: To use `'inferred'`, a trained GST model must be in `gst_updated/results/`. Models are provided.

### 4. Sweep Functionality

Test robot with controlled linear motion:

```python
# In crowd_nav/configs/config.py
robot.sweep = True
robot.sweep_start = -5
robot.sweep_stop = 5
robot.sweep_step = 0.5
robot.sweep_dir = 'x'  # or 'y'
robot.sweep_axes = 0   # 0 for x-axis, 1 for y-axis
```

### 5. Record Trajectories

Capture robot and human states for system identification:

```python
# In crowd_nav/configs/config.py
env.record = True
```

Outputs saved to training directory for analysis.

### 6. Advanced Visualization

Save episode frames for slideshow generation:

```python
# In test.py
test_args.save_slides = True
test_args.visualize = True
```

Frames saved to `trained_models/<model_dir>/test/`

**Note**: This significantly slows testing. Refer to the [save_slides branch](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph/tree/save_slides) for details.

---

## Architecture Details

### Attention-Based Interaction Graph

**Key Components:**

1. **Human-Human Attention** (`use_self_attn=True`)
   - Models interactions between pedestrians
   - Captures social forces and group dynamics

2. **Robot-Human Attention** (`use_hr_attn=True`)
   - Models robot awareness of each pedestrian
   - Enables socially-aware planning

3. **Trajectory Prediction**
   - Predicts `predict_steps` future frames
   - Prevents collisions with intended paths

### Network Architecture

- **Graph Neural Network**: Heterogeneous edges for different interaction types
- **Temporal Modeling**: LSTM/RNN cells for sequential decision making
- **Attention Mechanism**: Soft attention over agent interactions
- **Policy Network**: Actor-Critic PPO for RL

---

## Known Limitations

1. **Python & OS Support**: Tested on Ubuntu with Python 3.6+ and 3.8+. Other OS/versions untested.

2. **Hyperparameter Sensitivity**: Performance varies with hyperparameters and random seeds. Manual tuning may be needed to match paper results.

3. **Trajectory Prediction**: GST model must be trained separately. See [gst repo](https://github.com/tedhuang96/gst).

4. **Sim2Real Gap**: Real-world deployment requires domain adaptation. Uncertainties in real environments may affect performance. See [Sim2Real tutorial](https://github.com/Shuijing725/CrowdNav_Sim2Real_Turtlebot) for guidance.

---

## Future Work & Extensions

- Multi-robot scenarios
- Hierarchical navigation with semantic understanding
- Integration with vision-based perception
- Deployment on additional robot platforms
- Adaptive prediction model selection

---

## Sim2Real Transfer

We provide a complete **Sim2Real transfer tutorial**:

📖 [CrowdNav Sim2Real Tutorial](https://github.com/Shuijing725/CrowdNav_Sim2Real_Turtlebot)

This includes:
- System identification for TurtleBot 2i
- Domain adaptation strategies
- Real-world deployment guidelines
- Troubleshooting common issues

---

## Citation

If you use this code or paper in your research, please cite:

```bibtex
@inproceedings{liu2022intention,
  title={Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph},
  author={Liu, Shuijing and Chang, Peixin and Huang, Zhe and Chakraborty, Neeloy and Hong, Kaiwen and Liang, Weihang and Livingston McPherson, D. and Geng, Junyi and Driggs-Campbell, Katherine},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023},
  pages={12015--12021}
}

@inproceedings{liu2020decentralized,
  title={Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning},
  author={Liu, Shuijing and Chang, Peixin and Liang, Weihang and Chakraborty, Neeloy and Driggs-Campbell, Katherine},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021},
  pages={3517--3524}
}
```

---

## Related Works

This repository builds on:

1. **Decentralized Structural-RNN** for crowd navigation  
   [Liu et al., ICRA 2021] | [GitHub](https://github.com/Shuijing725/CrowdNav_DSRNN)

2. **Gumbel Social Transformer** for trajectory prediction  
   [Huang et al., RA-L 2022] | [GitHub](https://github.com/tedhuang96/gst)

---

## Contributors

**Authors:**
- [Shuijing Liu](https://github.com/Shuijing725)

**Contributors:**
- [Peixin Chang](https://github.com/PeixinC)
- [Zhe Huang](https://github.com/tedhuang96)
- [Neeloy Chakraborty](https://github.com/TheNeeloy)

---

## License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

---

## Contact & Support

- 📧 **Issues**: Please open a [GitHub Issue](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph/issues)
- 🔧 **Pull Requests**: Contributions welcome!
- ❓ **Questions**: Feel free to ask in Issues

---

## Acknowledgments

- ICRA reviewers for constructive feedback
- University collaborators for simulation infrastructure
- Open-source community for PyTorch, OpenAI Gym, and other tools

---

**Last Updated**: June 2024 | **PyTorch Version**: 1.12.1 | **Python**: 3.8+
