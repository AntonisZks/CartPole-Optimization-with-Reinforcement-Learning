<div align="center">
    <h1><b>CartPole Environment with Reinforcement Learning</b></h1>
</div>

<div align="center">

![License](https://img.shields.io/github/license/AntonisZks/Reinforcement-Learning-Assignment.svg)
![Repository Size](https://img.shields.io/github/repo-size/AntonisZks/Reinforcement-Learning-Assignment.svg)
![Release](https://img.shields.io/github/v/release/AntonisZks/Reinforcement-Learning-Assignment.svg)
![Contributors](https://img.shields.io/github/contributors/AntonisZks/Reinforcement-Learning-Assignment.svg)
	
</div>

## 📋 Table of Contents
- [Team Members](#team-members)
- [Project Overview](#project-overview)
- [Environment Analysis](#environment-analysis)
- [Implementation](#implementation)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Models and Architectures](#models-and-architectures)
- [Contributing](#contributing)

## 👥 Team Members

<div align="center">

| First Name | Last Name | Student ID | Email |
|:---:|:---:|:---:|:---:|
| Antonis | Zikas | 1115202100038 | sdi2100038@di.uoa.gr |
| Panagiotis | Papapostolou | 1115202100142 | sdi2100142@di.uoa.gr |

</div>

## 🎯 Project Overview

This project implements and experiments with various **Reinforcement Learning algorithms** to train agents on the **CartPole-v1** environment from OpenAI Gymnasium. The main focus is on **Deep Q-Network (DQN)** implementations with different architectural variations and comparative analysis with other RL algorithms.

### Key Features
- 🧠 **Multiple DQN Implementations**: Standard DQN, Dueling Architecture, and Transformer-based Q-Networks
- 📊 **Comprehensive Analysis**: Performance comparison with random actions and sensitivity studies
- 🔧 **Modular Design**: Clean, well-documented code structure
- 📈 **Visualization**: Detailed plotting and analysis of training results
- 🎮 **Environment Testing**: Baseline performance analysis with random actions
- 🏆 **State-of-the-art Algorithms**: Integration with Stable-Baselines3 (PPO, A2C)

## 🎮 Environment Analysis

### CartPole-v1 Environment
The CartPole-v1 environment is a classic control problem where the goal is to balance a pole on a cart by moving the cart left or right.

#### Action Space
- **Discrete**: 2 possible actions
  - `0`: Move cart to the left
  - `1`: Move cart to the right

#### Observation Space
- **Continuous**: 4-dimensional state vector
  - `[0]`: Cart Position (range: -4.8 to 4.8)
  - `[1]`: Cart Velocity (range: -∞ to +∞)
  - `[2]`: Pole Angle (range: ~-0.418 to 0.418 radians)
  - `[3]`: Pole Angular Velocity (range: -∞ to +∞)

#### Reward System
- **+1** for each timestep the pole remains upright
- **Reward threshold**: 500 (considered solved)
- Episode terminates when pole angle > ±12° or cart position > ±2.4

## 🚀 Implementation

### Core Algorithms Implemented

1. **Deep Q-Network (DQN)**
   - Experience replay buffer
   - Target network for stable learning
   - ε-greedy exploration strategy

2. **Dueling Architecture DQN**
   - Separate value and advantage streams
   - Improved learning efficiency

3. **Transformer-based Q-Network**
   - Sequential state processing
   - Attention mechanism for temporal dependencies

4. **Stable-Baselines3 Integration**
   - Proximal Policy Optimization (PPO)
   - Advantage Actor-Critic (A2C)

### Key Components

- **Neural Networks**: Fully connected layers with ReLU activations
- **Replay Buffer**: Experience replay for stable training  
- **Target Networks**: Periodic updates for learning stability
- **Exploration Strategy**: ε-greedy with exponential decay

## 📁 Project Structure

```
Reinforcement-Learning-Assignment/
│
├── notebooks/
│   └── cart_pole.ipynb          # Main Jupyter notebook with experiments
│
├── src/
│   ├── agents.py                # DQN Agent implementation
│   ├── networks.py              # Neural network architectures
│   ├── trainers.py              # Training logic and utilities
│   ├── replay_buffers.py        # Experience replay buffer
│   ├── testing.py               # Model testing and evaluation
│   ├── plotting.py              # Visualization utilities
│   ├── utils.py                 # Helper functions and hyperparameters
│   ├── dqn.py                   # Main DQN training script
│   ├── env_showcase.py          # Environment demonstration
│   ├── stable_baselines_a2c.py  # A2C training with Stable-Baselines3
│   └── stable_baselines_ppo.py  # PPO training with Stable-Baselines3
│
├── models/                      # Saved trained models
│   ├── dqn_model.pth
│   ├── dueling_arc_dqn_model.pth
│   ├── transformer_model.pth
│   ├── ppo_*.pth
│   └── a2c_*.pth
│
├── reports/
│   ├── figs/                    # Generated plots and visualizations
│   └── PDFs/                    # Final report documents
│
├── logs/
│   └── tensorboard/             # TensorBoard logging for training metrics
│
├── assets/
│   └── imgs/                    # Images and diagrams
│
├── docs/                        # Assignment documentation
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Reinforcement-Learning-Assignment
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Key Dependencies
- `torch`: Deep learning framework
- `gymnasium`: OpenAI Gym environments
- `stable-baselines3`: State-of-the-art RL algorithms
- `matplotlib`: Plotting and visualization
- `numpy`: Numerical computations
- `jupyter`: Interactive notebooks

## 🎮 Usage

### Quick Start

1. **Run Environment Showcase**
   ```bash
   python src/env_showcase.py
   ```

2. **Train DQN Agent**
   ```bash
   python src/dqn.py
   ```

3. **Train with Stable-Baselines3**
   ```bash
   python src/stable_baselines_ppo.py  # For PPO
   python src/stable_baselines_a2c.py  # For A2C
   ```

4. **Interactive Analysis**
   ```bash
   jupyter notebook notebooks/cart_pole.ipynb
   ```

### Hyperparameter Configuration

Key hyperparameters are defined in `src/utils.py`:

```python
GAMMA = 0.99          # Discount factor
LR = 1e-3             # Learning rate
BATCH_SIZE = 64       # Minibatch size
MEMORY_SIZE = 10000   # Replay buffer size
EPSILON_START = 1.0   # Starting exploration probability
EPSILON_END = 0.01    # Minimum exploration probability
EPSILON_DECAY = 0.995 # Epsilon decay rate
TARGET_UPDATE = 10    # Target network update frequency
```

## 📊 Results

### Performance Comparison

| Algorithm | Average Score | Success Rate | Training Episodes |
|-----------|---------------|--------------|------------------|
| Random Actions | ~22 | ~10% | N/A |
| DQN | ~475+ | ~95%+ | 500 |
| Dueling DQN | ~480+ | ~96%+ | 500 |
| Transformer DQN | ~450+ | ~90%+ | 500 |
| PPO (Stable-Baselines3) | ~500 | ~99% | Variable |
| A2C (Stable-Baselines3) | ~495+ | ~98% | Variable |

### Key Findings
- ✅ All implemented algorithms significantly outperform random actions
- ✅ Dueling architecture shows slight improvement over standard DQN
- ✅ Stable-Baselines3 implementations achieve near-optimal performance
- ✅ Transformer-based approach shows promise but requires tuning

## 🧠 Models and Architectures

### Standard DQN Architecture
```
Input Layer (4 nodes) → Hidden Layer (128) → Hidden Layer (128) → Hidden Layer (128) → Output Layer (2 nodes)
```

### Dueling Architecture
- **Shared layers**: 4 → 128 → 128 → 128 → 128
- **Value stream**: 128 → 64 → 1
- **Advantage stream**: 128 → 64 → 2
- **Combination**: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))

### Transformer Architecture
- **Sequence length**: 10 timesteps
- **Embedding dimension**: 64
- **Attention heads**: 4
- **Encoder layers**: 2

## 📈 Monitoring and Visualization

The project includes comprehensive visualization tools:

- **Training Progress**: Score and epsilon decay over episodes
- **Performance Comparison**: Trained agents vs. random actions
- **Sensitivity Analysis**: Hyperparameter impact studies
- **TensorBoard Integration**: Real-time training metrics

## 🤝 Contributing

This is an academic project for coursework. The implementation follows best practices for:

- **Code Organization**: Modular, well-documented structure
- **Reproducibility**: Seed setting for consistent results
- **Experimentation**: Comprehensive sensitivity studies
- **Visualization**: Clear, informative plots and metrics

---

<div align="center">
    <p><em>This project is part of the coursework for Reinforcement Learning & Stochastic Games</em></p>
    <p><strong>National and Kapodistrian University of Athens</strong></p>
    <p>Department of Informatics and Telecommunications</p>
</div>