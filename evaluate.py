import argparse
import os
import time

import numpy as np
import torch

from commanded_bioloid_env import CommandedBioloidEnv
from sac_agent import soft_actor_critic_agent

COMMAND_NAMES = { 0:"Stop", 1: "Forward", 2: "Backward", 3: "Left", 4: "Right"}

def evaluate_policy(agent, env, episodes=3):
    """Runs evaluation episodes for each command and prints the results."""
    for cmd_id, cmd_name in COMMAND_NAMES.items():
        total_return = 0.0
        print(f"--- Evaluating command: {cmd_name} ---")
        for i in range(episodes):
            state, _ = env.reset()
            env.set_command(cmd_id)
            state = env._get_obs()
            
            done = False
            ep_ret = 0.0
            ep_steps = 0
            while not done:
                action = agent.select_action(state, eval=True)
                state, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                ep_ret += float(reward)
                ep_steps += 1
                # Slow down the simulation to make it watchable
                time.sleep(1./240.)
            
            total_return += ep_ret
            print(f"  Run {i+1}/{episodes}: Return = {ep_ret:.2f}, Steps = {ep_steps}")
        
        avg_return = total_return / episodes
        print(f"-> Average Return for '{cmd_name}': {avg_return:.2f}\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC agent for the Bioloid robot.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model checkpoint (e.g., dir_bioloid_ant_like/checkpoint_ep1000.pth)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes for each command.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize environment in GUI mode to watch the agent
    env = CommandedBioloidEnv(render_mode="GUI")
    # It's good practice to seed the environment's random number generator as well
    # This was missing in the original training script's evaluation function
    env.np_random.seed(args.seed) 

    # Initialize agent
    agent = soft_actor_critic_agent(
        env.observation_space.shape[0],
        env.action_space,
        device=device,
        hidden_size=256,
        seed=args.seed,
        lr=5e-4,  # Learning rate, not used in eval but needed for init
        gamma=0.99, # Discount factor, not used in eval but needed for init
        tau=0.005,   # Soft update coefficient, not used in eval but needed for init
        alpha=0.2, # Temperature parameter, not used in eval but needed for init
    )

    # Load the trained model
    if not os.path.isfile(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        env.close()
        return
        
    data = torch.load(args.model_path, map_location=device)
    if "actor" in data:
        agent.policy.load_state_dict(data['actor'])
    agent.policy.eval() # Set the actor to evaluation mode
    print(f"Loaded actor weights from: {args.model_path}")

    print("\nStarting evaluation...")
    evaluate_policy(agent, env, args.episodes)
    print("Evaluation finished.")

    env.close()

if __name__ == "__main__":
    main()