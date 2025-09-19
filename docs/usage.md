# Space Combat RL - Usage Guide

This guide covers how to use the unified training system for behavior cloning pretraining and reinforcement learning.

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Overview](#system-overview)
3. [Data Collection](#data-collection)
4. [Behavior Cloning (BC) Pretraining](#behavior-cloning-bc-pretraining)
5. [Reinforcement Learning Training](#reinforcement-learning-training)
6. [Model Evaluation](#model-evaluation)
7. [Human Play Testing](#human-play-testing)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

## Quick Start

### Complete Pipeline (Recommended)
```bash
# 1. Collect training data (scripted vs scripted)
python src/collect_data.py collect_bc --config src/unified_training.yaml

# 2. Run full pipeline: BC pretraining → RL training → evaluation
python src/unified_train.py full --config src/unified_training.yaml

# 3. Test the final model
python src/collect_data.py evaluate_model --model checkpoints/unified_full_*/final_rl_model.zip
```

### Human Play (Test Environment)
```bash
# Play against scripted agents to test the environment
python src/collect_data.py human_play
```

## System Overview

The unified system supports multiple agent types and training modes:

- **Scripted Agents**: Your sophisticated predictive targeting agents
- **RL Models**: Transformer-based or PPO models
- **Human Players**: Human input via keyboard/mouse
- **Self-Play**: Models playing against previous versions

### Key Components

```
src/
├── agents.py              # Pluggable agent system
├── game_runner.py         # Universal game runner
├── rl_wrapper.py          # SB3-compatible RL wrapper
├── collect_data.py        # Data collection & evaluation
├── unified_train.py       # Training pipeline
├── bc_training.py         # BC pretraining module
├── unified_training.yaml  # Configuration
└── [existing files...]    # Your current src files
```

## Data Collection

### BC Training Data (Scripted vs Scripted)

Collect episodes of scripted agents playing against each other:

```bash
# Basic collection (uses config defaults)
python src/collect_data.py collect_bc

# With custom config
python src/collect_data.py collect_bc --config src/unified_training.yaml

# Custom output directory
python src/collect_data.py collect_bc --output data/my_bc_data
```

**Default Collection:**
- 2,500 episodes each of 1v1, 2v2, 3v3, 4v4 (10,000 total)
- ~1-2M state-action pairs
- Includes Monte Carlo returns for value training
- Saves compressed pickle files

**Output:**
```
data/bc_pretraining_TIMESTAMP/
├── bc_training_data.pkl.gz    # All episodes combined
├── 1v1_episodes.pkl.gz        # Episodes by game mode
├── 2v2_episodes.pkl.gz
├── 3v3_episodes.pkl.gz
├── 4v4_episodes.pkl.gz
└── collection_stats.yaml      # Statistics
```

### Self-Play Data Collection

Collect data from trained models playing against each other:

```bash
# Requires trained models
python src/collect_data.py collect_selfplay --config src/selfplay_config.yaml
```

## Behavior Cloning (BC) Pretraining

Train a policy to imitate scripted agent behavior:

### Standalone BC Training
```bash
# Train BC model (requires collected data)
python src/unified_train.py bc --config src/unified_training.yaml
```

### BC Training Details

**What it does:**
- Loads collected scripted vs scripted episodes
- Trains transformer with policy + value heads
- Policy learns to predict scripted agent actions
- Value network learns to predict episode returns

**Output:**
```
checkpoints/unified_bc_TIMESTAMP/
├── best_bc_model.pt           # Best model (lowest validation loss)
├── final_bc_model.pt          # Final epoch model
├── bc_model_epoch_*.pt        # Periodic checkpoints
├── training_curves.png        # Loss curves
└── config.yaml               # Configuration copy
```

**Training Progress:**
- Policy loss: Binary cross-entropy on action prediction
- Value loss: MSE on Monte Carlo returns
- Early stopping based on validation loss
- Typical training: 20-50 epochs, ~2-4 hours

## Reinforcement Learning Training

Train agents using PPO with various opponent types:

### RL Training Modes

**From Scratch:**
```bash
python src/unified_train.py rl --config src/unified_training.yaml
```

**From BC Model (Recommended):**
```bash
python src/unified_train.py rl --config src/unified_training.yaml \
    --bc-model checkpoints/unified_bc_*/best_bc_model.pt
```

**Full Pipeline (BC → RL):**
```bash
python src/unified_train.py full --config src/unified_training.yaml
```

### Opponent Types

Configure in `src/unified_training.yaml`:

```yaml
training:
  rl:
    opponent:
      type: "mixed"              # scripted, self_play, mixed
      scripted_mix_ratio: 0.3    # 30% scripted, 70% self-play
```

- **scripted**: Always vs scripted agents (stable, good for early training)
- **self_play**: Always vs previous model versions (challenging, good for late training)
- **mixed**: Randomly choose scripted or self-play each episode (balanced)

### Training Output

```
checkpoints/unified_rl_TIMESTAMP/
├── final_rl_model.zip         # Final PPO model
├── best_model.zip             # Best evaluation model
├── rl_model_*_steps.zip       # Periodic checkpoints
└── config.yaml               # Configuration copy

logs/unified_rl_TIMESTAMP/
├── train/                     # Training logs
├── eval/                      # Evaluation logs
└── progress.csv               # Training progress
```

### Training Progress

Monitor training with:
- **Win rate**: Against scripted opponents
- **Episode length**: Shorter = more decisive victories
- **Reward**: Team rewards per episode
- **Self-play memory**: Number of stored opponent models

## Model Evaluation

### Evaluate Single Model

```bash
# Evaluate transformer model
python src/collect_data.py evaluate_model \
    --model checkpoints/unified_bc_*/best_bc_model.pt \
    --config src/unified_training.yaml

# Evaluate PPO model
python src/collect_data.py evaluate_model \
    --model checkpoints/unified_rl_*/final_rl_model.zip \
    --config src/unified_training.yaml
```

### Evaluation Output

```
==========================================================
EVALUATION RESULTS
==========================================================
Model: checkpoints/unified_bc_20240320_143022/best_bc_model.pt
Episodes: 100
Game mode: 2v2
Wins: 45
Losses: 50
Draws: 5
Win Rate: 47.4%
Average Episode Length: 127.3
```

### Interpreting Results

- **Win Rate vs Scripted**: ~50% = well-trained BC model, >60% = strong RL model
- **Win Rate vs Random**: Should be >90% for any decent model
- **Episode Length**: Shorter often indicates more decisive play

## Human Play Testing

Test the environment and agents interactively:

```bash
python src/collect_data.py human_play
```

### Controls
- **Movement**: WASD or Arrow Keys
- **Shoot**: Spacebar
- **Sharp Turn**: Shift (more agile but uses more energy)
- **Quit**: Close window or Ctrl+C

### Human Play Features
- Real-time combat against scripted agents
- Win/loss tracking and statistics
- Multiple game modes (1v1, 2v2, etc.)
- Visual feedback on ship health and power

## Configuration

### Main Config File: `src/unified_training.yaml`

Key sections:

```yaml
# Environment settings
environment:
  world_size: [1200, 800]
  max_ships: 8

# Data collection
data_collection:
  bc_data:
    episodes_per_mode:
      "1v1": 2500
      "2v2": 2500
      "3v3": 2500  
      "4v4": 2500

# BC training
model:
  bc:
    learning_rate: 0.001
    batch_size: 128
    epochs: 50

# RL training  
training:
  rl:
    total_timesteps: 2000000
    opponent:
      type: "mixed"
      scripted_mix_ratio: 0.3
```

### Custom Configurations

Create custom configs for different experiments:

```bash
# Custom BC training
cp src/unified_training.yaml src/my_bc_config.yaml
# Edit my_bc_config.yaml
python src/unified_train.py bc --config src/my_bc_config.yaml

# Custom RL training
cp src/unified_training.yaml src/my_rl_config.yaml  
# Edit my_rl_config.yaml
python src/unified_train.py rl --config src/my_rl_config.yaml
```

## Advanced Usage

### Custom Data Collection

```bash
# Collect specific amount of data
python src/collect_data.py collect_bc --output data/small_dataset \
    --config src/small_data_config.yaml

# Collect self-play data from multiple models
python src/collect_data.py collect_selfplay \
    --config src/selfplay_config.yaml
```

### Model Comparison

```bash
# Evaluate multiple models
for model in checkpoints/*/best_*.pt; do
    echo "Evaluating $model"
    python src/collect_data.py evaluate_model --model "$model"
done
```

### Resume Training

```bash
# Resume RL training from checkpoint
python src/unified_train.py rl --config src/unified_training.yaml \
    --bc-model checkpoints/interrupted_rl_model.zip
```

## File Organization

After running the full pipeline:

```
project/
├── src/
│   ├── collect_data.py         # Data collection script
│   ├── unified_train.py        # Training pipeline
│   ├── unified_training.yaml   # Main config
│   ├── agents.py              # Agent system
│   ├── game_runner.py         # Game runner
│   ├── rl_wrapper.py          # RL wrapper
│   ├── bc_training.py         # BC training
│   └── [existing files...]    # Your current files
├── checkpoints/
│   ├── unified_bc_*/          # BC models
│   └── unified_rl_*/          # RL models  
├── logs/
│   ├── unified_bc_*/          # BC training logs
│   └── unified_rl_*/          # RL training logs
├── data/
│   └── bc_pretraining_*/      # BC training data
└── docs/
    └── USAGE.md               # This file
```

## Troubleshooting

### Common Issues

**"No BC training data found"**
```bash
# Solution: Collect data first
python src/collect_data.py collect_bc
```

**"CUDA out of memory"**
```yaml
# Solution: Reduce batch size in config
model:
  bc:
    batch_size: 64  # Reduce from 128
  ppo:
    batch_size: 64  # Reduce from 128
```

**"Module not found errors"**
```bash
# Solution: Run from project root directory
cd /path/to/project
python src/collect_data.py collect_bc

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/project/src"
```

**"Model evaluation fails"**
```bash
# Check model type matches evaluation config
python src/collect_data.py evaluate_model --model path/to/model \
    --config src/evaluation_config.yaml
```

**"Human play window doesn't open"**
```bash
# Install pygame if missing
pip install pygame
```

### Performance Tips

**Faster Data Collection:**
- Reduce episodes in config for testing
- Use compressed storage (enabled by default)
- Run on multiple cores if available

**Faster Training:**
- Use GPU for model training
- Reduce model size for experimentation
- Use smaller batch sizes if memory constrained

**Better Models:**
- Collect more diverse training data
- Use BC pretraining before RL
- Tune opponent mix ratios
- Increase training time

### Debugging

**Enable verbose logging:**
```yaml
logging:
  console:
    level: "DEBUG"
```

**Monitor training:**
```bash
# Watch training progress
tail -f logs/unified_*/train/monitor.csv

# Check model checkpoints
ls -la checkpoints/unified_*/
```

**Test individual components:**
```bash
# Test scripted agents
python src/collect_data.py human_play

# Test data loading
python -c "
import sys; sys.path.append('src')
from bc_training import BCDataset
BCDataset(['data/bc_*/bc_training_data.pkl.gz'], 0)
"

# Test model creation
python -c "
import sys; sys.path.append('src')
from bc_training import create_bc_model
create_bc_model({'embed_dim': 64})
"
```

### Import Issues

Since all files are in `src/`, you may need to handle imports:

**Option 1: Run from project root**
```bash
# Always run commands from project root
cd /path/to/project
python src/collect_data.py collect_bc
```

**Option 2: Add src to Python path**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/collect_data.py collect_bc
```

**Option 3: Use Python module syntax**
```bash
python -m src.collect_data collect_bc
```

## Next Steps

1. **Start with data collection** to understand the system
   ```bash
   python src/collect_data.py collect_bc
   ```

2. **Try human play** to test the environment
   ```bash
   python src/collect_data.py human_play
   ```

3. **Run BC pretraining** to get baseline models
   ```bash
   python src/unified_train.py bc
   ```

4. **Experiment with RL training** using different opponents
   ```bash
   python src/unified_train.py rl --bc-model checkpoints/unified_bc_*/best_bc_model.pt
   ```

5. **Evaluate and compare** different approaches
   ```bash
   python src/collect_data.py evaluate_model --model checkpoints/unified_*/final_*.pt
   ```

For more detailed technical information, see the code documentation in each module.