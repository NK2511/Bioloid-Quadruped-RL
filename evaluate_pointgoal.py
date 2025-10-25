import os
import argparse
import numpy as np
import torch

from PointGoalBioloidEnv import PointGoalBioloidEnv
from sac_agent import soft_actor_critic_agent
import pybullet as p


def load_actor_from_checkpoint(agent, ckpt_path: str, device: torch.device):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    data = torch.load(ckpt_path, map_location=device)
    if "actor" not in data:
        raise KeyError(f"Checkpoint does not contain 'actor' weights: {ckpt_path}")
    agent.policy.load_state_dict(data["actor"])  # actor only is sufficient for evaluation
    agent.policy.eval()


def evaluate(episodes: int, checkpoint: str, render: bool = False, seed: int = 0, deterministic: bool = True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create env
    env = PointGoalBioloidEnv(render_mode="GUI" if render else "DIRECT")

    # Build agent with correct dims
    agent = soft_actor_critic_agent(
        env.observation_space.shape[0],
        env.action_space,
        device=device,
        hidden_size=256,
        seed=seed,
        lr=5e-4,  # lr is irrelevant at eval time
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
    )

    load_actor_from_checkpoint(agent, checkpoint, device)

    returns = []
    lengths = []
    successes = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0
        goal_reached = False

        # --- Visualize goal in GUI: green sphere ---
        try:
            # Remove previous goal marker if any
            if hasattr(evaluate, "_goal_vis_id") and evaluate._goal_vis_id is not None:
                try:
                    p.removeBody(evaluate._goal_vis_id)
                except Exception:
                    pass
        except Exception:
            pass
        goal_xy = np.array(env.target_goal, dtype=np.float32)
        goal_pos = [float(goal_xy[0]), float(goal_xy[1]), 0.02]
        try:
            vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0.1, 0.9, 0.1, 1.0])
            evaluate._goal_vis_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_shape, basePosition=goal_pos)
        except Exception:
            evaluate._goal_vis_id = None
        evaluate._line_uid = None

        while not done:
            with torch.no_grad():
                action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            # Draw/refresh a line from robot base to goal for visualization
            try:
                base_pos, _ = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
                start = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                end = [goal_pos[0], goal_pos[1], 0.02]
                if getattr(evaluate, "_line_uid", None) is not None:
                    try:
                        p.removeUserDebugItem(evaluate._line_uid)
                    except Exception:
                        pass
                evaluate._line_uid = p.addUserDebugLine(start, end, [0.1, 0.9, 0.1], lineWidth=2.0, lifeTime=0)
            except Exception:
                pass

            done = bool(terminated or truncated)
            ep_ret += float(reward)
            ep_len += 1
            if info.get("goal_reached", False):
                goal_reached = True

        returns.append(ep_ret)
        lengths.append(ep_len)
        successes.append(1.0 if goal_reached else 0.0)
        print(f"Episode {ep:03d} | Return: {ep_ret:8.2f} | Len: {ep_len:4d} | Success: {goal_reached}")

    returns = np.array(returns, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.int32)
    successes = np.array(successes, dtype=np.float32)

    print("\n=== Evaluation Summary ===")
    print(f"Episodes:         {episodes}")
    print(f"Average Return:   {returns.mean():.2f} ± {returns.std():.2f}")
    print(f"Average Length:   {lengths.mean():.1f}")
    print(f"Success Rate:     {successes.mean()*100:.1f}%")

    env.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to full checkpoint .pth containing actor")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--render", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stochastic", action="store_true", help="(unused) reserved flag; agent API does not expose deterministic eval")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        episodes=args.episodes,
        checkpoint=args.checkpoint,
        render=args.render,
        seed=args.seed,
        deterministic=not args.stochastic,
    )
