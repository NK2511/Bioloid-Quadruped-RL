import os
import time
import argparse
from collections import deque
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from PointGoalBioloidEnv import PointGoalBioloidEnv
from sac_agent import soft_actor_critic_agent
from replay_memory import ReplayMemory

class colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RED = '\033[91m'
    PURPLE = '\033[95m'
    BOLD = '\033[1m'
    END = '\033[0m'

def save_full_checkpoint(agent, directory: str, episode: int, total_steps: int, updates: int, extra: Optional[Dict[str, Any]] = None) -> str:
    os.makedirs(directory, exist_ok=True)
    ckpt = {
        "actor": agent.policy.state_dict(),
        "critic": agent.critic.state_dict(),
        "policy_optim": getattr(agent, "policy_optim", None).state_dict() if getattr(agent, "policy_optim", None) else None,
        "critic_optim": getattr(agent, "critic_optim", None).state_dict() if getattr(agent, "critic_optim", None) else None,
        "alpha": float(getattr(agent, "alpha", torch.tensor(0.0)).detach().cpu().item()),
        "log_alpha": getattr(agent, "log_alpha", None).detach().cpu() if getattr(agent, "log_alpha", None) is not None else None,
        "alpha_optim": getattr(agent, "alpha_optim", None).state_dict() if getattr(agent, "alpha_optim", None) else None,
        "episode": int(episode),
        "total_steps": int(total_steps),
        "updates": int(updates),
    }
    if extra:
        ckpt.update(extra)
    path = os.path.join(directory, f"checkpoint_ep{episode}.pth")
    torch.save(ckpt, path)
    return path

def load_full_resume(agent, resume_full_path: str, device: torch.device, is_transfer_learning: bool = False) -> Dict[str, int]:
    info = {"episode": 0, "total_steps": 0, "updates": 0}
    if not (resume_full_path and os.path.isfile(resume_full_path)):
        return info
    
    data = torch.load(resume_full_path, map_location=device)
    
    def transfer_weights(new_model, old_state_dict):
        new_state_dict = new_model.state_dict()
        print("\n[Transfer] Attempting to transfer weights...")
        for name, param in new_state_dict.items():
            if name in old_state_dict and old_state_dict[name].shape == param.shape:
                param.copy_(old_state_dict[name])
                print(f"  - [OK] Copied layer: {name}")
            else:
                print(f"  - [SKIP] Mismatch or missing layer: {name}")
        new_model.load_state_dict(new_state_dict)
        print("[Transfer] Weight transfer complete.\n")

    if is_transfer_learning:
        if data.get("actor"):
            transfer_weights(agent.policy, data["actor"])
        if data.get("critic"):
            transfer_weights(agent.critic, data["critic"])
            with torch.no_grad():
                agent.critic_target.load_state_dict(agent.critic.state_dict())
        print(f"[Resume] Transfer-loaded model weights from {resume_full_path}")
        return info

    if data.get("actor"):
        agent.policy.load_state_dict(data["actor"])
    if data.get("critic"):
        agent.critic.load_state_dict(data["critic"])
        with torch.no_grad():
            agent.critic_target.load_state_dict(agent.critic.state_dict())
    if data.get("policy_optim") and getattr(agent, "policy_optim", None):
        agent.policy_optim.load_state_dict(data["policy_optim"])
    if data.get("critic_optim") and getattr(agent, "critic_optim", None):
        agent.critic_optim.load_state_dict(data["critic_optim"])
    if data.get("log_alpha") is not None and getattr(agent, "log_alpha", None) is not None:
        agent.log_alpha.data = data["log_alpha"].to(device)
        agent.alpha = agent.log_alpha.exp()
    if data.get("alpha_optim") and getattr(agent, "alpha_optim", None):
        agent.alpha_optim.load_state_dict(data["alpha_optim"])
    
    info["episode"] = int(data.get("episode", 0))
    info["total_steps"] = int(data.get("total_steps", 0))
    info["updates"] = int(data.get("updates", 0))
    print(f"[Resume] Loaded full checkpoint from {resume_full_path}")
    return info

class Callback:
    def on_training_start(self, ctx: Dict[str, Any]): pass
    def on_episode_start(self, ctx: Dict[str, Any]): pass
    def on_step(self, ctx: Dict[str, Any]): pass
    def on_episode_end(self, ctx: Dict[str, Any]): pass
    def on_training_end(self, ctx: Dict[str, Any]): pass

class TensorBoardCallback(Callback):
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_step(self, ctx: Dict[str, Any]):
        t = ctx["global_step"]
        self.writer.add_scalar("train/step_reward", ctx["reward"], t)
        # Only log scalar reward components to avoid TensorBoard shape errors
        if "info" in ctx:
            for key in ("reward_dist", "reward_ctrl"):
                if key in ctx["info"]:
                    try:
                        self.writer.add_scalar(f"reward_components/{key}", float(ctx["info"][key]), t)
                    except Exception:
                        pass

    def on_episode_end(self, ctx: Dict[str, Any]):
        ep = ctx["episode"]
        self.writer.add_scalar("train/episode_return", ctx["episode_reward"], ep)
        self.writer.add_scalar("train/episode_length", ctx["episode_steps"], ep)
        self.writer.add_scalar("train/avg100_return", ctx["avg_score"], ep)
        
        if "episode_reward_breakdown" in ctx:
            for key, value in ctx["episode_reward_breakdown"].items():
                self.writer.add_scalar(f"episode_reward_components/{key}", value, ep)

    def on_training_end(self, ctx: Dict[str, Any]):
        self.writer.close()

def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = PointGoalBioloidEnv(
        render_mode="DIRECT",
        max_torque=args.max_torque,
        ctrl_cost_weight=args.ctrl_cost_weight,
    )

    max_steps = env.max_episode_steps

    agent = soft_actor_critic_agent(
        env.observation_space.shape[0],
        env.action_space,
        device=device,
        hidden_size=256,
        seed=args.seed,
        lr=args.lr,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
    )

    memory = ReplayMemory(args.seed, 1_000_000)

    print("device:", device)
    print("state dim:", env.observation_space.shape[0])
    print("action space:", env.action_space)
    print("learning rate:", args.lr)

    resume_info = {"episode": 0, "total_steps": 0, "updates": 0}
    if args.resume_full_path:
        resume_info = load_full_resume(agent, args.resume_full_path, device, is_transfer_learning=args.transfer)
    start_episode = int(resume_info.get("episode", 0)) + (1 if resume_info.get("episode", 0) > 0 else 0)
    
    callbacks = [TensorBoardCallback(args.log_dir)]

    sac_train(
        env,
        agent,
        memory,
        batch_size=args.batch_size,
        start_steps=args.start_steps,
        num_episodes=args.num_episodes,
        max_steps=max_steps,
        callbacks=callbacks,
        gradient_updates_per_step=args.gradient_updates_per_step,
        save_dir=args.save_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume_episode=start_episode,
    )

    env.close()

def sac_train(env, agent, memory, batch_size, start_steps, num_episodes, max_steps, callbacks: List[Callback], gradient_updates_per_step: int = 1, save_dir="dir_pointgoal_bioloid", checkpoint_interval=0, resume_episode=0):
    total_numsteps = 0
    updates = 0
    time_start = time.time()
    scores_deque = deque(maxlen=100)

    for cb in callbacks: cb.on_training_start({"env": env, "agent": agent})

    for i_episode in range(resume_episode, num_episodes):
        state, _ = env.reset()
        for cb in callbacks: cb.on_episode_start({"episode": i_episode, "env": env})

        episode_reward = 0.0
        episode_steps = 0
        # Modified reward breakdown for the new environment
        episode_reward_breakdown = {
            "reward_dist": 0.0,
            "reward_ctrl": 0.0,
        }

        for step in range(max_steps):
            if total_numsteps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # This will now correctly log the new info keys
            for key, value in info.items():
                if key in episode_reward_breakdown:
                    episode_reward_breakdown[key] += value

            memory.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += float(reward)
            episode_steps += 1
            total_numsteps += 1

            for cb in callbacks:
                cb.on_step({"global_step": total_numsteps, "reward": float(reward), "info": info})

            if len(memory) > batch_size:
                for _ in range(gradient_updates_per_step):
                    agent.update_parameters(memory, batch_size, updates)
                    updates += 1

            if done:
                break

        scores_deque.append(episode_reward)
        avg_score = float(np.mean(scores_deque))

        for cb in callbacks:
            cb.on_episode_end({
                "episode": i_episode,
                "episode_reward": episode_reward,
                "episode_steps": episode_steps,
                "avg_score": avg_score,
                "episode_reward_breakdown": episode_reward_breakdown,
            })

        s = int(time.time() - time_start)
        reward_str = " | ".join([f"{colors.GREEN if 'dist' in k else colors.RED}{k}: {v:.2f}{colors.END}" for k,v in episode_reward_breakdown.items()])
        print(f"Ep: {i_episode} | Steps: {episode_steps} | {colors.BOLD}Score: {episode_reward:.2f}{colors.END} | Avg: {avg_score:.2f} | Time: {s//3600:02}:{(s%3600)//60:02}:{s%60:02}")
        print(f"   └ {reward_str}")

        if checkpoint_interval and (i_episode % checkpoint_interval == 0) and i_episode > 0:
            path = save_full_checkpoint(agent, save_dir, i_episode, total_numsteps, updates)
            print(f"[Checkpoint] Saved full checkpoint: {path}")

    for cb in callbacks: cb.on_training_end({})

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", type=str, default="dir_pointgoal_bioloid")
    p.add_argument("--log-dir", type=str, default="runs/pointgoal_sac")
    p.add_argument("--checkpoint-interval", type=int, default=100)
    p.add_argument("--resume-full-path", type=str, default="")
    p.add_argument("--transfer", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-episodes", type=int, default=100_000)
    p.add_argument("--gradient-updates-per-step", type=int, default=2)
    # Robot control parameters
    p.add_argument("--max-torque", type=float, default=1.0, help="Max per-joint torque (N·m) applied in TORQUE_CONTROL")
    p.add_argument("--ctrl-cost-weight", type=float, default=0.01, help="Weight for control penalty in reward")
    return p.parse_args()

if __name__ == "__main__":
    main()
