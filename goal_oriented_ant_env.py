
import gymnasium as gym
import numpy as np
from gymnasium.core import Wrapper

class GoalOrientedAntEnv(Wrapper):
    """
    A wrapper for the Ant-v5 environment that adds a goal-reaching task.
    The observation space is modified to include the goal coordinates, and the
    reward function is changed to incentivize moving towards the goal.
    """
    def __init__(self, render_mode=None):
        # Initialize the underlying Ant-v5 environment
        super().__init__(gym.make("Ant-v5", render_mode=render_mode))
        
        # Original observation space
        original_obs_space = self.env.observation_space
        
        # Define the bounds for the goal coordinates (x, y)
        # Let's assume the goal can be within a 10x10 area around the origin
        goal_low = np.array([-10.0, -10.0])
        goal_high = np.array([10.0, 10.0])
        
        # Define the new observation space by appending the goal space
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([original_obs_space.low, goal_low]),
            high=np.concatenate([original_obs_space.high, goal_high]),
            dtype=np.float64
        )
        
        self.goal = None
        self.prev_xy = None  # track previous xy position for potential-based reward
        self.goal_radius = 10.0 # Radius within which to sample goals
        self.goal_threshold = 0.5 # Distance to consider the goal reached

    def reset(self, **kwargs):
        """
        Resets the environment and samples a new random goal.
        """
        obs, info = self.env.reset(**kwargs)
        
        # Sample a new goal uniformly within the goal_radius
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.sqrt(np.random.uniform(0, self.goal_radius**2))
        self.goal = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        
        # Initialize previous position from info if available, else origin
        if isinstance(info, dict) and "x_position" in info and "y_position" in info:
            self.prev_xy = np.array([info["x_position"], info["y_position"]], dtype=float)
        else:
            self.prev_xy = np.array([0.0, 0.0], dtype=float)
        
        # Append the goal to the observation
        modified_obs = np.concatenate([obs, self.goal])
        return modified_obs, info

    def step(self, action):
        """
        Takes a step in the environment, calculates the goal-oriented reward,
        and returns the modified observation.
        """
        obs, _, terminated, truncated, info = self.env.step(action)

        # Determine new xy position from info if available
        if isinstance(info, dict) and "x_position" in info and "y_position" in info:
            new_xy = np.array([info["x_position"], info["y_position"]], dtype=float)
        else:
            new_xy = self.prev_xy if self.prev_xy is not None else np.array([0.0, 0.0], dtype=float)
        old_xy = self.prev_xy if self.prev_xy is not None else new_xy

        # --- Reward Calculation (potential-based) ---
        distance_before = np.linalg.norm(old_xy - self.goal)
        distance_after = np.linalg.norm(new_xy - self.goal)
        reward = float(distance_before - distance_after)

        # Control effort penalty if available
        try:
            ctrl_cost = float(self.env.unwrapped.control_cost(action))
        except Exception:
            ctrl_cost = 0.0
        reward -= ctrl_cost

        # Check if the goal is reached
        goal_reached = bool(distance_after < self.goal_threshold)
        if goal_reached:
            reward += 100.0
            terminated = True

        # Update prev for next step
        self.prev_xy = new_xy

        # Enrich info
        if isinstance(info, dict):
            info.update({
                "goal": self.goal.copy(),
                "distance_after": float(distance_after),
                "distance_before": float(distance_before),
                "goal_reached": goal_reached,
                "ctrl_cost": -float(ctrl_cost),
            })

        # Append the goal to the observation
        modified_obs = np.concatenate([obs, self.goal])
        
        return modified_obs, reward, terminated, truncated, info

def main():
    """
    An example of how to use the GoalOrientedAntEnv.
    The Ant will take random actions.
    """
    # You might need to install gymnasium and mujoco first:
    # pip install gymnasium[mujoco]
    
    env = GoalOrientedAntEnv(render_mode='human')
    
    for episode in range(5):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        
        print(f"--- Episode {episode + 1}, Goal: {env.goal} ---")
        
        while not terminated and not truncated:
            action = env.action_space.sample() # Take a random action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # The render call is handled by the wrapper
            env.render()

        print(f"Episode finished. Total reward: {episode_reward}")

    env.close()

if __name__ == "__main__":
    main()
