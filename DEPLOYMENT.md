# Jetbot Policy Deployment Guide

## Overview
This guide helps you deploy your Isaac Lab trained policy to the Nvidia Jetbot hardware.

## Prerequisites
- Nvidia Jetbot with JetPack installed
- PyTorch installed on Jetbot (usually pre-installed with JetPack)
- SSH access to your Jetbot

## Setup

### 1. Transfer Files to Jetbot
```bash
# From your development machine
scp jetbot_policy.pt jetbot@<jetbot-ip>:~/
scp run_policy.py jetbot@<jetbot-ip>:~/
scp requirements.txt jetbot@<jetbot-ip>:~/
```

### 2. Install Dependencies on Jetbot
```bash
# SSH into Jetbot
ssh jetbot@<jetbot-ip>

# Install required packages
pip3 install -r requirements.txt
```

### 3. Inspect Your Policy (IMPORTANT)
Before configuring observations, find out what your policy expects:

```bash
python3 inspect_policy.py
```

This will show you:
- Required input dimensions (e.g., "Input size 14 works!")
- Output dimensions (should match your action space)
- Output value ranges

**Example output:**
```
✓ Input size 14 works!
  Input shape:  torch.Size([1, 14])
  Output shape: torch.Size([1, 2])
  Output range: [-0.234, 0.456]
```

This tells you your policy needs 14 observation values and outputs 2 action values.

### 4. Configure Observation Space
Edit `run_policy.py` and update the `get_observation()` method to match your Isaac Lab training.

**Why this matters:** Your policy was trained on specific sensor data in Isaac Lab. The real robot must provide the exact same type and number of observations, or the policy won't work.

**Step-by-step:**

1. **Find your Isaac Lab observation definition** - Check your training config/code for something like:
```python
# Example from Isaac Lab training
observation_space = {
    "base_lin_vel": 3,      # x, y, z linear velocity
    "base_ang_vel": 3,      # roll, pitch, yaw angular velocity
    "projected_gravity": 3, # gravity vector
    "commands": 3,          # target velocities
    "actions": 2,           # previous motor commands
}
# Total: 14 values
```

2. **Implement the same observations on Jetbot:**
```python
def get_observation(self):
    obs = []

    # Base velocities (from encoders or estimated from motor commands)
    obs.extend([0.0, 0.0, 0.0])  # lin_vel x,y,z
    obs.extend([0.0, 0.0, 0.0])  # ang_vel roll,pitch,yaw

    # Gravity direction (from IMU if available)
    # imu = self.robot.imu.read()
    # obs.extend(imu['gravity'])
    obs.extend([0.0, 0.0, -9.81])  # Default: gravity pointing down

    # Command/goal (what you want the robot to do)
    obs.extend([0.5, 0.0, 0.0])  # e.g., move forward at 0.5 m/s

    # Previous actions (track in your class)
    obs.extend([self.last_left_speed, self.last_right_speed])

    return torch.FloatTensor(obs).unsqueeze(0).to(self.device)
```

**Common observation sources:**
- **Motor encoders**: Current wheel velocities
- **IMU sensor**: Orientation, angular velocity, linear acceleration
- **Previous actions**: What commands you sent last timestep
- **Command/goal**: Target behavior (forward speed, turn rate, etc.)
- **Estimated base velocity**: Calculate from wheel speeds

**If you don't have some sensors:**
- Start with zeros or reasonable estimates
- Focus on the most important observations first
- Your policy may still work with approximate data

### 5. Configure Action Space
Update the `process_action()` method to match your action space:

```python
def process_action(self, action):
    # If your action is differential drive: [left_wheel, right_wheel]
    left_speed = float(action[0, 0].item()) * self.max_speed
    right_speed = float(action[0, 1].item()) * self.max_speed

    # Or if your action is [linear_vel, angular_vel]:
    # linear_vel = float(action[0, 0].item())
    # angular_vel = float(action[0, 1].item())
    # left_speed = (linear_vel - angular_vel * wheel_base / 2) * scale
    # right_speed = (linear_vel + angular_vel * wheel_base / 2) * scale

    return left_speed, right_speed
```

## Running the Policy

### Basic Usage
```bash
# Run with default settings (50Hz, max speed 0.3)
python3 run_policy.py

# Run with custom settings
python3 run_policy.py --freq 30 --max-speed 0.5

# Run for specific duration (in seconds)
python3 run_policy.py --duration 60

# Use different policy file
python3 run_policy.py --policy path/to/policy.pt
```

### Command Line Arguments
- `--policy`: Path to policy file (default: `jetbot_policy.pt`)
- `--freq`: Control loop frequency in Hz (default: 50)
- `--max-speed`: Maximum motor speed 0-1 (default: 0.3)
- `--duration`: Run duration in seconds (default: infinite)

## Safety Tips

1. **Start with low speed**: Use `--max-speed 0.2` initially
2. **Test in safe area**: Clear space free of obstacles
3. **Have killswitch ready**: Keep finger on Ctrl+C or have physical access to stop the robot
4. **Check control frequency**: Monitor for "Warning: Loop taking..." messages
5. **Gradual speed increase**: Only increase speed after verifying stable behavior

## Troubleshooting

### Policy loads but robot doesn't move
- Check observation dimensions match your training
- Verify policy is outputting reasonable actions (add debug prints)
- Ensure motors are enabled and battery charged

### Control loop too slow
- Reduce control frequency: `--freq 20`
- Check CPU usage on Jetbot
- Consider model optimization (TorchScript is already used)

### Robot behavior differs from simulation
- **Observation mismatch**: Ensure real sensors match simulation data format
- **Action scaling**: Your policy may output actions in different range than expected
- **Domain randomization**: May need to add noise/variation in training
- **Sensor noise**: Real sensors are noisier than simulation

### Understanding Observation Space Mismatch

**The most common deployment issue** is observation space mismatch. Here's what happens:

**Symptom:** Policy loads successfully but robot behaves erratically or doesn't move properly.

**Cause:** The real robot is providing different observations than what the policy was trained on.

**How to diagnose:**
1. Run `inspect_policy.py` to find required input size
2. Check your Isaac Lab training config for observation definitions
3. Verify `get_observation()` returns the correct number of values
4. Ensure values are in the same range/units as training

**Example mismatch:**
```python
# Training used: [lin_vel(3), ang_vel(3), gravity(3), commands(3), actions(2)] = 14 values
# But real robot only provides: [commands(3), actions(2)] = 5 values
# Result: Policy receives wrong input → bad behavior
```

**Debugging tips:**
```python
# Add to get_observation() temporarily
obs = torch.FloatTensor(obs_list).unsqueeze(0).to(self.device)
print(f"Observation shape: {obs.shape}, values: {obs}")
return obs
```

### Common Observation/Action Mismatches
```python
# Check your policy I/O shapes
print(f"Policy input shape: {obs.shape}")  # Should match Isaac Lab obs space
print(f"Policy output shape: {action.shape}")  # Should match action space
```

## Next Steps

### Add Sensors
If your policy needs additional observations:
```python
# IMU example (if jetbot has IMU)
from adafruit_lsm9ds1 import LSM9DS1_I2C
imu = LSM9DS1_I2C(i2c)
orientation = imu.acceleration  # or gyro, magnetometer
```

### Add Vision (Future)
For vision-based policies:
- Use Jetbot camera: `from jetbot import Camera`
- Preprocess images to match training
- Consider ONNX+TensorRT for speed

### ROS 2 Integration (Future)
If you need ROS 2 ecosystem:
```bash
# Install ROS 2 Humble on Jetson
# Create ros2 package with policy node
# Subscribe to sensor topics, publish cmd_vel
```

## Performance Optimization

If inference is too slow:
1. Check GPU usage: `nvidia-smi`
2. Ensure policy is on CUDA: Script already does this
3. Consider ONNX + TensorRT export for 2-5x speedup
4. Reduce model size in training

## Questions?
Common issues and questions:
- **What observation space?**: Check your Isaac Lab config files
- **What action space?**: Check your training environment definition
- **Sim-to-real gap?**: May need domain randomization in training or system identification
