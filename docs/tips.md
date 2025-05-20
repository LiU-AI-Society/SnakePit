# 🎓 Training Tips & Best Practices

## 📚 Curriculum Learning

### 1. Progressive Training Path
1. **Basic Training** (500-1000 episodes)
   ```bash
   # Start with single snake 
   python train.py --episodes 500 
   ```

2. **Intermediate Training** (1000-2000 episodes)
   ```bash
   # Add basic opponents
   python train.py --episodes 1000 --opponents random_snake
   ```

3. **Advanced Training** (2000+ episodes)
   ```bash
   # Multiple opponents
   python train.py --episodes 2000 --opponents greedy_snake smart_snake 
   ```

Remember to update rewards between sessions

### 2. Performance Monitoring

#### Key Metrics to Watch
- Average episode length
- Win rate against different opponents
- Food collection rate
- Death reason distribution


#### Visualization Tools
- Use `training_stats.png` for trend analysis
- Watch live gameplay with visualization
- Monitor death reasons histogram

### 3. Common Issues & Solutions

#### Overfitting
**Symptoms:**
- Good performance against training opponents
- Poor performance against new opponents

**Solutions:**
- Rotate opponent types during training

#### Unstable Learning
**Symptoms:**
- Highly variable rewards
- Inconsistent behavior

**Solutions:**
```python
# Adjust these hyperparameters in train.py
GAMMA = 0.99        # Increase for more stable learning
LR = 1e-4          # Decrease if learning is unstable
PPO_CLIP = 0.1     # Reduce for more conservative updates
```

#### Poor Exploration
**Symptoms:**
- Snake gets stuck in repetitive patterns
- Fails to find food efficiently

**Solutions:**
```python
# Modify rewards.py for better exploration
self.reward_dict.update({
    "another_turn": 0.02,     # Increase survival reward
    "ate_food": 1.5,          # Boost food reward
    "moved_to_food": 0.1      # Add reward for moving toward food
})
```

### 4. Training Duration Guidelines

| Stage | Episodes | Focus Areas |
|-------|----------|-------------|
| Testing | 100-500 | Basic movement, reward tuning |
| Basic | 1000-2000 | Food collection, survival |
| Intermediate | 2000-5000 | Combat, efficiency |
| Advanced | 5000+ | Strategy refinement |

### 5. Save & Compare Models

```bash
# Save checkpoints during training
python train.py --save-interval 200

# Evaluate different models
python evaluate.py --model-1 runs/run01/ppo_final.pt --model-2 runs/run02/ppo_final.pt
```

### 6. Environment Variation Tips

- **Map Rotation:**
  ```bash
  python train.py --map-size 7 7 --episodes 500
  python train.py --map-size 11 11 --episodes 500
  ```

- **Opponent Mixing:**
  ```bash
  python train.py --opponents random_snake greedy_snake smart_snake
  ```

### 7. Debug Checklist

- [ ] Monitor reward components individually
- [ ] Check death reason distribution
- [ ] Verify observation space correctness
- [ ] Test against specific scenarios
- [ ] Analyze win rate trends

### 8. Competition Preparation

1. **Model Selection**
   - Choose best performing checkpoint
   - Test against diverse opponents
   - Verify stability across different maps

2. **Deployment**
   ```bash
   # Convert to ONNX for deployment
   python convert_to_onnx.py --model runs/best_run/ppo_final.pt
   
   # Test server locally
   python launch.py --model runs/best_run/final.onnx
   ```

3. **Final Checks**
   - Response time < 500ms
   - Memory usage within limits
   - Stable behavior in long games