#!/usr/bin/env python3
"""
Deploy Isaac Lab trained policy on Jetbot hardware.
Motor control policy without vision.
"""

import torch
import time
import numpy as np
from jetbot import Robot
import argparse


class JetbotPolicyRunner:
    def __init__(self, policy_path, control_freq=50, max_speed=0.3, goal_x=1.0, goal_y=0.0):
        """
        Initialize the policy runner.

        Args:
            policy_path: Path to the .pt policy file
            control_freq: Control loop frequency in Hz
            max_speed: Maximum motor speed (0-1 range)
            goal_x: Goal position X in meters (relative to start position)
            goal_y: Goal position Y in meters (relative to start position)
        """
        print(f"Loading policy from {policy_path}...")
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()

        # Initialize robot
        print("Initializing Jetbot...")
        self.robot = Robot()

        # Move to GPU if available (Jetson should have CUDA)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)
        print(f"Running on device: {self.device}")

        self.control_freq = control_freq
        self.max_speed = max_speed
        self.dt = 1.0 / control_freq

        # State tracking for observations
        self.left_wheel_pos = 0.0   # Accumulated wheel position (radians)
        self.right_wheel_pos = 0.0
        self.left_wheel_vel = 0.0   # Current wheel velocity (rad/s)
        self.right_wheel_vel = 0.0
        self.robot_x = 0.0          # Estimated robot position (meters)
        self.robot_y = 0.0
        self.robot_yaw = 0.0        # Estimated robot orientation (radians)
        self.lin_vel_x = 0.0        # Linear velocity in world frame (m/s)
        self.lin_vel_y = 0.0
        self.ang_vel_z = 0.0        # Angular velocity (rad/s)

        # Goal tracking (relative to current position)
        self.goal_x = goal_x
        self.goal_y = goal_y

        # Robot parameters (Jetbot specific - adjust if needed)
        self.wheel_radius = 0.03    # meters (30mm wheels)
        self.wheel_base = 0.10      # meters (100mm between wheels)

    def get_observation(self):
        """
        Collect observations from the robot.

        TODO: Adapt this to match your Isaac Lab observation space.
        Common observations for motor policies:
        - Motor velocities
        - IMU data (orientation, angular velocity, linear acceleration)
        - Previous actions
        - Target velocity/direction

        Returns:
            torch.Tensor: Observation vector matching policy input
        """
        # Observation space from Isaac Lab training (9 values):
        # [lin_vel_x, lin_vel_y, ang_vel_z, left_wheel_pos, right_wheel_pos,
        #  left_wheel_vel, right_wheel_vel, goal_rel_x, goal_rel_y]

        # Calculate relative goal position (goal - current position)
        goal_rel_x = self.goal_x - self.robot_x
        goal_rel_y = self.goal_y - self.robot_y

        obs = np.array([
            self.lin_vel_x,         # 0: Linear velocity X in world frame (m/s)
            self.lin_vel_y,         # 1: Linear velocity Y in world frame (m/s)
            self.ang_vel_z,         # 2: Angular velocity around Z-axis (rad/s)
            self.left_wheel_pos,    # 3: Left wheel joint position (radians)
            self.right_wheel_pos,   # 4: Right wheel joint position (radians)
            self.left_wheel_vel,    # 5: Left wheel joint velocity (rad/s)
            self.right_wheel_vel,   # 6: Right wheel joint velocity (rad/s)
            goal_rel_x,             # 7: Relative goal position X (meters)
            goal_rel_y,             # 8: Relative goal position Y (meters)
        ], dtype=np.float32)

        return torch.FloatTensor(obs).unsqueeze(0).to(self.device)

    def update_state_from_motors(self, left_speed, right_speed):
        """
        Update robot state estimates based on motor commands.
        This is a simple odometry estimate - not perfect but workable.

        Args:
            left_speed: Left motor speed (-1 to 1)
            right_speed: Right motor speed (-1 to 1)
        """
        # Convert motor speeds to wheel velocities (rad/s)
        # Assuming motor speed maps to wheel angular velocity
        # This is an approximation - tune based on your robot
        max_wheel_vel = 10.0  # rad/s (adjust based on your motors)
        self.left_wheel_vel = left_speed * max_wheel_vel
        self.right_wheel_vel = right_speed * max_wheel_vel

        # Update wheel positions (integrate velocity)
        self.left_wheel_pos += self.left_wheel_vel * self.dt
        self.right_wheel_pos += self.right_wheel_vel * self.dt

        # Calculate linear and angular velocities from wheel velocities
        # Differential drive kinematics
        v_left = self.left_wheel_vel * self.wheel_radius
        v_right = self.right_wheel_vel * self.wheel_radius

        linear_vel = (v_left + v_right) / 2.0
        self.ang_vel_z = (v_right - v_left) / self.wheel_base

        # Update robot pose (dead reckoning)
        self.robot_yaw += self.ang_vel_z * self.dt
        self.robot_x += linear_vel * np.cos(self.robot_yaw) * self.dt
        self.robot_y += linear_vel * np.sin(self.robot_yaw) * self.dt

        # Linear velocities in world frame
        self.lin_vel_x = linear_vel * np.cos(self.robot_yaw)
        self.lin_vel_y = linear_vel * np.sin(self.robot_yaw)

    def process_action(self, action):
        """
        Convert policy action to motor commands.

        Args:
            action: torch.Tensor from policy output

        Returns:
            tuple: (left_speed, right_speed) in range [-max_speed, max_speed]
        """
        # Policy outputs [left_motor, right_motor]
        left_speed = float(action[0, 0].item()) * self.max_speed
        right_speed = float(action[0, 1].item()) * self.max_speed

        # Clamp to safe range
        left_speed = np.clip(left_speed, -self.max_speed, self.max_speed)
        right_speed = np.clip(right_speed, -self.max_speed, self.max_speed)

        return left_speed, right_speed

    def run(self, duration=None):
        """
        Run the policy control loop.

        Args:
            duration: Optional duration in seconds. If None, runs until interrupted.
        """
        print(f"Starting policy at {self.control_freq} Hz (Ctrl+C to stop)")
        print(f"Max speed: {self.max_speed}")

        start_time = time.time()
        iteration = 0

        try:
            while True:
                loop_start = time.time()

                # Get observation
                obs = self.get_observation()

                # Run policy inference
                with torch.no_grad():
                    action = self.policy(obs)

                # Convert to motor commands
                left_speed, right_speed = self.process_action(action)

                # Send to motors
                self.robot.set_motors(left_speed, right_speed)

                # Update state estimate for next observation
                self.update_state_from_motors(left_speed, right_speed)

                # Print status every second
                if iteration % self.control_freq == 0:
                    elapsed = time.time() - start_time
                    dist_to_goal = np.sqrt((self.goal_x - self.robot_x)**2 + (self.goal_y - self.robot_y)**2)
                    print(f"[{elapsed:.1f}s] L: {left_speed:+.3f}, R: {right_speed:+.3f} | "
                          f"Pos: ({self.robot_x:.2f}, {self.robot_y:.2f}) | Goal dist: {dist_to_goal:.2f}m")

                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    print(f"Duration {duration}s reached")
                    break

                # Maintain control frequency
                elapsed = time.time() - loop_start
                if elapsed < self.dt:
                    time.sleep(self.dt - elapsed)
                elif elapsed > self.dt * 1.5:
                    print(f"Warning: Loop taking {elapsed*1000:.1f}ms (target: {self.dt*1000:.1f}ms)")

                iteration += 1

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.robot.stop()
            print("Robot stopped")


def main():
    parser = argparse.ArgumentParser(description="Run Isaac Lab policy on Jetbot")
    parser.add_argument("--policy", type=str, default="jetbot_policy.pt",
                        help="Path to policy file")
    parser.add_argument("--freq", type=int, default=50,
                        help="Control frequency in Hz")
    parser.add_argument("--max-speed", type=float, default=0.3,
                        help="Maximum motor speed (0-1)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Run duration in seconds (default: infinite)")
    parser.add_argument("--goal-x", type=float, default=1.0,
                        help="Goal position X in meters (default: 1.0)")
    parser.add_argument("--goal-y", type=float, default=0.0,
                        help="Goal position Y in meters (default: 0.0)")

    args = parser.parse_args()

    runner = JetbotPolicyRunner(
        policy_path=args.policy,
        control_freq=args.freq,
        max_speed=args.max_speed,
        goal_x=args.goal_x,
        goal_y=args.goal_y
    )

    runner.run(duration=args.duration)


if __name__ == "__main__":
    main()
