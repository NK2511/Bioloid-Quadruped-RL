import os
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data

class PointGoalBioloidEnv(gym.Env):
    """
    A goal-reaching environment for the Bioloid robot using PyBullet.
    The reward structure is designed to be similar to the successful GoalOrientedAntEnv.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self, render_mode=None, max_torque: float = 1.0, ctrl_cost_weight: float = 0.01):
        self.render_mode = render_mode
        
        # --- PyBullet Setup ---
        if self.render_mode == 'human':
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        self.client_id = self.client  # alias used by evaluation scripts
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # --- Robot and World Loading ---
        self.plane_id = p.loadURDF("plane.urdf")
        # Use the absolute path provided by the user
        urdf_path = r"C:\Users\nandh\Downloads\Bioloid_Quadruped_Model\Bioloid_Quadruped_Model.urdf"
        self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.25])

        # --- Robot Joint and Link Info ---
        self.motor_joint_indices = [0, 1, 2, 3, 4, 5, 6, 7]
        self.n_motors = len(self.motor_joint_indices)
        self.torso_link_index = -1 # -1 refers to the base
        self.max_torque = float(max_torque) # Max torque to apply to motors
        self.ctrl_cost_weight = float(ctrl_cost_weight) # Penalty for motor usage

        # --- Goal and Episode Parameters ---
        self.goal = None
        self.target_goal = None  # alias for external code
        self.goal_radius = 5.0
        self.goal_threshold = 0.2
        self.max_episode_steps = 1000
        self.current_step = 0

        # RNG for goal sampling
        self.np_random = np.random.RandomState()

        # --- Action and Observation Spaces ---
        # Actions are normalized motor torques [-1, 1]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.n_motors,), dtype=np.float32)

        # Observations: base_pos(z), base_orientation(quat), joint_pos, joint_vel, goal_coords
        obs_dim = 1 + 4 + self.n_motors + self.n_motors + 2
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _get_obs(self):
        """Constructs the observation array for the agent."""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        joint_states = p.getJointStates(self.robot_id, self.motor_joint_indices, physicsClientId=self.client)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]

        goal_xy = self.goal if self.goal is not None else np.array([0.0, 0.0])
        obs = np.concatenate([
            [pos[2]], # z-position of the base
            orn,       # base orientation as a quaternion
            joint_pos,
            joint_vel,
            goal_xy # goal coordinates
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random.seed(seed)
        self.current_step = 0
        
        # Reset robot state
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0.25], [0, 0, 0, 1], physicsClientId=self.client)
        for i in self.motor_joint_indices:
            p.resetJointState(self.robot_id, i, targetValue=0, targetVelocity=0, physicsClientId=self.client)

        # Sample a new goal
        angle = self.np_random.uniform(0, 2 * np.pi)
        radius = np.sqrt(self.np_random.uniform(0, self.goal_radius**2))
        self.goal = np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32)
        self.target_goal = self.goal.copy()

        info = {"goal": self.goal.copy()}
        return self._get_obs(), info

    def step(self, action):
        old_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        
        # Apply action as motor torques
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.motor_joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=(np.asarray(action, dtype=np.float32) * self.max_torque).tolist(),
            physicsClientId=self.client,
        )

        p.stepSimulation(physicsClientId=self.client)
        self.current_step += 1

        new_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        
        # --- Reward Calculation (mirrors GoalOrientedAntEnv) ---
        distance_before = np.linalg.norm(np.array(old_pos[:2]) - self.goal)
        distance_after = np.linalg.norm(np.array(new_pos[:2]) - self.goal)
        
        reward = float(distance_before - distance_after)
        
        # Penalty for motor usage
        action_arr = np.asarray(action, dtype=np.float32)
        ctrl_cost = float(self.ctrl_cost_weight * np.square(action_arr).mean())
        reward -= ctrl_cost

        # --- Termination Conditions ---
        terminated = False
        goal_reached = False
        # 1. Goal reached
        if distance_after < self.goal_threshold:
            goal_reached = True
            reward += 100.0
            terminated = True

        # 2. Robot has fallen over
        if new_pos[2] < 0.08:
            reward -= 100.0
            terminated = True

        # 3. Episode timeout
        truncated = self.current_step >= self.max_episode_steps

        info = {
            'reward_dist': float(distance_before - distance_after),
            'reward_ctrl': -ctrl_cost,
            'goal_reached': goal_reached,
            'goal': self.goal.copy(),
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        # This is handled by the GUI connection in __init__
        pass

    def close(self):
        if self.client >= 0:
            p.disconnect(self.client)
            self.client = -1

# --- Example Usage --- 
if __name__ == '__main__':
    env = PointGoalBioloidEnv(render_mode='human')
    for episode in range(3):
        obs, info = env.reset()
        terminated, truncated = False, False
        episode_reward = 0
        print(f"--- Episode {episode + 1}, Goal: {env.goal} ---")
        while not terminated and not truncated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
        print(f"Episode finished. Total reward: {episode_reward:.2f}")
    env.close()
