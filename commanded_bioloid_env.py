# --- commanded_bioloid_env.py ---

import math
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pybullet as p
from gymnasium import spaces

from bioloid_ant_env import BioloidAntLikeEnv

class CommandedBioloidEnv(BioloidAntLikeEnv):
    """
    Environment where the policy learns to execute discrete commands:
    0: Stop
    1: Move Forward
    2: Move Backward
    3: Move Left
    4: Move Right
    """

    def __init__(self, *args, **kwargs):
        self.fixed_speed = kwargs.pop('fixed_speed', 0.3)
        self.w_tracking = kwargs.pop('w_tracking', 1.0)
        self.w_ortho_penalty = kwargs.pop('w_ortho_penalty', 0.1)

        # Define the discrete commands and their target velocities [vx, vy]
        self.command_map = {
            0: np.array([0.0, 0.0], dtype=np.float32),               # Stop
            1: np.array([self.fixed_speed, 0.0], dtype=np.float32),  # Forward
            2: np.array([-self.fixed_speed, 0.0], dtype=np.float32), # Backward
            3: np.array([0.0, self.fixed_speed], dtype=np.float32),   # Left
            4: np.array([0.0, -self.fixed_speed], dtype=np.float32),  # Right
        }
        self.num_commands = len(self.command_map)
        self.command = 0  # Current command

        super().__init__(*args, **kwargs)

        # Augment observation space with one-hot encoded command
        base_obs_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(base_obs_dim + self.num_commands,), dtype=np.float32
        )

    def _sample_command(self):
        """Generates a new random discrete command for training.""" 
        self.command = self.np_random.randint(0, self.num_commands)

    def set_command(self, command_id: int):
        """Manually sets the command for evaluation."""
        if command_id in self.command_map:
            self.command = command_id
        else:
            raise ValueError(f"Invalid command_id: {command_id}")

    def _get_obs(self) -> np.ndarray:
        base_obs = super()._get_obs()
        one_hot_command = np.eye(self.num_commands, dtype=np.float32)[self.command]
        return np.concatenate([base_obs, one_hot_command])

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        _, info = super().reset(seed=seed, options=options)
        self._sample_command()
        return self._get_obs(), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        target_positions = []
        for i, (low, high) in enumerate(self.joint_limits):
            target_positions.append(np.interp(action[i], [-1, 1], [low, high]))

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
            forces=[self.torque_limit] * len(self.joint_indices),  # Max force for each joint
            physicsClientId=self.client_id,
        )

        for _ in range(self.frame_skip):
            p.stepSimulation(physicsClientId=self.client_id)

        obs = self._get_obs()
        self.step_count += 1
        
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)
        vx, vy, _ = base_lin_vel
        _, _, wz = base_ang_vel
        
        target_vel = self.command_map[self.command]
        target_vx, target_vy = target_vel

        # Reward for matching the target velocity vector
        vel_error = np.sqrt((vx - target_vx)**2 + (vy - target_vy)**2)
        tracking_reward = self.w_tracking * math.exp(-2.0 * vel_error)

        # Penalty for unwanted rotation
        wz_penalty = self.w_ortho_penalty * wz**2

        ctrl_cost = self.ctrl_cost_weight * float(np.sum(np.square(action)))
        contact_cost = self.contact_cost_weight * self._sum_contact_forces()

        total_reward = self.alive_bonus + tracking_reward - ctrl_cost - contact_cost - wz_penalty

        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps
        
        info: Dict[str, Any] = {
            "command": self.command,
            "target_vx": target_vx,
            "target_vy": target_vy,
            "actual_vx": vx,
            "actual_vy": vy,
            "alive_bonus": self.alive_bonus,
            "tracking_reward": tracking_reward,
            "wz_penalty": -wz_penalty,
            "ctrl_cost": -ctrl_cost,
            "contact_cost": -contact_cost,
        }

        return obs, float(total_reward), bool(terminated), bool(truncated), info
