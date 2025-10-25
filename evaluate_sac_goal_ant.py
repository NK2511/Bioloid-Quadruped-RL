import os
import torch
import numpy as np
import time
import mujoco

from goal_oriented_ant_env import GoalOrientedAntEnv
from sac_agent import soft_actor_critic_agent
from train_ant_goal_sac import load_full_resume


def draw_goal_marker(env, goal_pos, color=(1.0, 0.0, 0.0, 1.0), radius=0.25, z_height=0.2):
    """
    Draw a sphere marker at the goal position in the MuJoCo viewer.
    Tries multiple viewer backends (Gymnasium mujoco renderer, classic viewer).
    """
    # Resolve a viewer from various wrappers
    viewer = None
    for candidate in (env, getattr(env, "env", None), getattr(env, "unwrapped", None)):
        if candidate is None:
            continue
        if hasattr(candidate, "mujoco_renderer") and getattr(candidate.mujoco_renderer, "viewer", None) is not None:
            viewer = candidate.mujoco_renderer.viewer  # Gymnasium mujoco renderer
            break
        if hasattr(candidate, "viewer") and candidate.viewer is not None:
            viewer = candidate.viewer  # Older viewer API
            break

    if viewer is None:
        # No viewer available (likely render_mode is not human)
        return

    try:
        marker = mujoco.MjvGeom()
        mujoco.mjv_defaultGeom(marker)
        marker.type = mujoco.mjtGeom.mjGEOM_SPHERE
        marker.size[:] = [float(radius), 0.0, 0.0]
        marker.pos[:] = [float(goal_pos[0]), float(goal_pos[1]), float(z_height)]
        marker.rgba[:] = color
        # Add to the current scene; scene is rebuilt each frame, so no manual cleanup needed.
        mujoco.mjv_addGeom(viewer.scn, marker)
    except Exception:
        # Best-effort only; ignore drawing errors on unsupported viewers
        pass


def evaluate_sac_model(checkpoint_path: str, num_episodes: int = 5, render: bool = True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hint MuJoCo to use a windowed backend so a viewer exists
    if render and not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "glfw"

    # Initialize environment
    env = GoalOrientedAntEnv(render_mode="human" if render else None)
    obs, _ = env.reset()

    # Force render to initialize the viewer (critical for MuJoCo)
    if render:
        env.render()
        print("Called env.render() to initialize viewer.")

    # Debug: Check if viewer is initialized (after render call)
    viewer_available = False
    if hasattr(env, "mujoco_renderer") and hasattr(env.mujoco_renderer, "viewer") and env.mujoco_renderer.viewer is not None:
        viewer_available = True
    elif hasattr(env, "viewer") and env.viewer is not None:
        viewer_available = True
    elif hasattr(env.unwrapped, "viewer") and env.unwrapped.viewer is not None:
        viewer_available = True
    print("Viewer initialized:", viewer_available)

    # Initialize SAC agent
    agent = soft_actor_critic_agent(
        env.observation_space.shape[0],
        env.action_space,
        device=device,
        hidden_size=256,
        seed=0,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
    )

    # Load checkpoint
    info = load_full_resume(agent, checkpoint_path, device)
    print(f"\n✅ Loaded model from {checkpoint_path}")
    print(f"Checkpoint info: {info}\n")

    agent.policy.eval()

    # Run evaluation episodes
    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False
        steps = 0

        goal = env.goal
        print(f"\n🎯 Episode {ep+1}/{num_episodes}")
        print(f"Goal position: {goal}")

        while not (terminated or truncated):
            with torch.no_grad():
                action = agent.select_action(obs, eval=True)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # Draw goal marker every frame with dynamic color
            if render:
                dist_to_goal = np.linalg.norm(obs[:2] - goal)
                color = (0.0, 1.0, 0.0, 1.0) if dist_to_goal < env.goal_threshold else (1.0, 0.0, 0.0, 1.0)
                draw_goal_marker(env, goal, color=color)
                
                # Explicitly render the viewer to update the scene
                env.render()

            if render:
                time.sleep(1 / 60.0)

        print(f"Episode {ep+1} finished in {steps} steps | Total Reward: {total_reward:.2f}")

    env.close()
    print("\n✅ Evaluation finished.")


if __name__ == "__main__":
    checkpoint = "dir_ant_goal_sac/checkpoint_ep3000.pth"
    evaluate_sac_model(checkpoint, num_episodes=5, render=True)
