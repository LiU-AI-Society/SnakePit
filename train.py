import os
import time
import json
import torch
import pygame
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

from Gym.snake_gym import BattlesnakeGym
from visualize import init_visualization, visualize_step, close_visualization, reset_sound_state_tracker, SNAKE_COLORS
from ppo import PPOAgent, PPOMemory, ActorCritic  # from your PPO module
from utils import get_next_run_dir, game_state_to_observation, load_opponent_modules, get_valid_actions
from utils import moving_average, plot_training_stats_live, plot_death_reasons, battle_snake_game_state_to_observation, game_state_to_matrix
from heuristic import heuristic
from utils import log_and_plot_training, determine_winner, plot_hyperparameters_and_rewards, debug_pause_step
from Gym.rewards import SimpleRewards


# ---- Config ----
TOTAL_TIMESTEPS = 1000_000_000     # Total environment steps to train for (outer loop limit)
SAVE_INTERVAL = 1000            # How often (episodes) to save model checkpoints
N_STEPS_COLLECT = 2000      # How many steps to collect before each PPO update (batch size for memory buffer)
PPO_BATCH_SIZE = 64             # Minibatch size for PPO update (used inside PPOAgent.learn)
N_EPOCHS_PPO = 4                # Number of epochs to iterate over the collected batch in each PPO update
LR = 3e-4                       # Learning rate for optimizer
GAMMA = 0.995                    # Discount factor for future rewards
GAE_LAMBDA = 0.95               # Lambda for Generalized Advantage Estimation (GAE)
PPO_CLIP = 0.2                  # PPO clipping parameter (epsilon)

YOUR_SNAKE_INDEX = 0
OBS_TYPE = "flat-51s"           # Observation type for the environment 5s for head and 1s for body
MAP_SIZE = (11, 11)


def train(
    total_timesteps=TOTAL_TIMESTEPS,  # Replace episodes with total_timesteps
    map_size=(11, 11),
    visualize=True,
    save_interval=100,
    visualize_interval=100,  
    opponent_names=None,
    pretrained=None,
    plot_every=100,
    our_snake_name="Our snake",
    continue_when_dead=False, 
    mute=False,              
    debug_mode=False,         
    visualization_fps=10):  # MODIFIED
    run_dir = get_next_run_dir()
    os.makedirs(run_dir, exist_ok=True)
    if debug_mode:
        visualize = True

    # Load opponents and names
    opponents = load_opponent_modules(opponent_names) # This now returns list of dicts with 'name'
    # Extract names for logging, using the 'name' field from opponent_config
    opponent_display_names = [opp_config["name"] for opp_config in opponents]
    snake_names = [our_snake_name] + opponent_display_names

    num_snakes = 1 + len(opponents)

    # Set up environment (no snake_names argument)
    env = BattlesnakeGym(
        observation_type=OBS_TYPE,
        map_size=map_size,  # <-- This is set by the --map-size argument
        number_of_snakes=num_snakes,
        verbose=False
    )
    obs_shape = env.observation_space.shape
    obs_dims = (3, obs_shape[0], obs_shape[1])
    action_dim = 4

    # Init PPO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(input_dims=obs_dims, n_actions=action_dim, lr=LR, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
                     ppo_clip=PPO_CLIP, n_epochs=N_EPOCHS_PPO, batch_size_ppo=PPO_BATCH_SIZE, device=device)

    if pretrained:
        agent.load_model(pretrained)
        agent.save_model(f"{run_dir}/init.pt")

    plot_hyperparameters_and_rewards(run_dir)


    memory = PPOMemory(batch_size=N_STEPS_COLLECT)

    # Visualization setup via visualize.py
    if visualize:
        screen, clock = init_visualization(map_size, fps=visualization_fps, mute=mute)  # MODIFIED
        # Try to use our_snake_name for index 0 in color mapping, fallback to board names
        try:
            snake_colors = {name: SNAKE_COLORS[i % len(SNAKE_COLORS)] for i, name in enumerate(snake_names)}
        except Exception:
            snake_colors = {s['name']: SNAKE_COLORS[i % len(SNAKE_COLORS)] for i, s in enumerate(env.get_json()['board']['snakes'])}

    episode_rewards, episode_lengths, death_reasons = [], [], []
    episode_apples = []  # new list to record apples eaten (snake length) per episode
    episode_wins_all = [[] for _ in range(num_snakes)]  # win flags for each snake
    obs_raw, _, _, info = env.reset()
    reset_sound_state_tracker()  # Reset sound tracker for first game state
    observation = game_state_to_observation(obs_raw)
    obs_tensor = torch.tensor(observation.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

    episode_reward, episode_len = 0, 0
    total_steps, episode_num = 0, 0
    json_obj = env.get_json()
    quit_training_flag = False

    while total_steps < total_timesteps and not quit_training_flag:  # Use total_timesteps here
        for step in range(N_STEPS_COLLECT):
            # Handle pygame
            if visualize:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        visualize = False
                        quit_training_flag = True # Ensure training loop also exits
                        break
                if not visualize: # Check again if QUIT event was processed
                    break

            # Debug mode pause before action selection
            if debug_mode and not quit_training_flag and visualize: # Added visualize check
                cont = debug_pause_step(
                    episode_num, step, total_steps,
                    json_obj, observation, obs_tensor,
                    agent, screen, clock, snake_colors,
                    snake_index=YOUR_SNAKE_INDEX,
                    mute=mute,
                    visualization_fps=visualization_fps  # MODIFIED
                )
                if not cont:
                    quit_training_flag = True
                    visualize = False # Ensure visualization stops if quit from debug
                    break
            
            if quit_training_flag: # Break from N_STEPS_COLLECT loop
                break

            current_snake_obj = env.snakes.get_snakes()[YOUR_SNAKE_INDEX]
            action_mask = None
            if current_snake_obj.is_alive() and len(current_snake_obj.locations) >= 2: # Only if snake has a neck
                valid_moves, moves, action_mask = heuristic(observation)
            action, log_prob, val = agent.choose_action(obs_tensor, action_mask=action_mask)

            actions = [action]
            # Get opponent actions
            for i, opponent_config in enumerate(opponents):
                opponent_fn = opponent_config["func"]
                is_opponent_model = opponent_config["is_model"]
                opponent_name = opponent_config["name"] # Get the display name
                opponent_snake_game_idx = i + 1 

                try:
                    is_snake_present_in_json = False
                    if opponent_snake_game_idx < len(json_obj['board']['snakes']):
                        opponent_snake_data_from_json = json_obj['board']['snakes'][opponent_snake_game_idx]
                        if opponent_snake_data_from_json.get('body'): # Check if body is not empty
                             is_snake_present_in_json = True
                    
                    if not is_snake_present_in_json:
                        actions.append(0)  # Dead or non-existent snake, default action
                        continue
                except (IndexError, KeyError):
                    actions.append(0) # Snake not found or error, default action
                    continue

                if is_opponent_model:
                    full_game_matrix = game_state_to_matrix(json_obj) 
                    # So, for opponent_snake_game_idx, its data is at channel (opponent_snake_game_idx + 1)
                    # game_state_to_observation's your_index is 0-based for snakes.
                    # So, if full_game_matrix is passed, and we want snake `k` (from json) to be 'you',
                    # then `your_index` for game_state_to_observation should be `k`.
                    # The `your_index` parameter in `game_state_to_observation` refers to the snake's index 
                    # within the `snakes` array of the `game_state` (or the equivalent channel index in `array_obs` 
                    # if we consider the food channel to be 0 and snake channels to start from 1).
                    # `opponent_snake_game_idx` is already the correct index for this purpose.
                    opponent_observation_for_model = game_state_to_observation(full_game_matrix, your_index=opponent_snake_game_idx)
                    
                    # Call the model opponent's action function
                    opponent_action = opponent_fn(opponent_observation_for_model, opponent_snake_game_idx)
                    actions.append(opponent_action)
                else:
                    opponent_action = opponent_fn(json_obj, opponent_snake_game_idx)
                    actions.append(opponent_action)

            next_raw_obs, rewards, dones, info = env.step(actions)
            next_json = env.get_json()

            next_obs = game_state_to_observation(next_raw_obs)
            next_tensor = torch.tensor(next_obs.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
            reward = rewards.get(0, 0)

            # Determine true game‐over
            alive = info.get('alive', {})
            alive_snakes = [i for i, flag in alive.items() if flag]
            our_dead = not alive.get(YOUR_SNAKE_INDEX, False)

            game_done = False # Initialize to False

            # Condition 1: Our snake is dead and we are not supposed to continue.
            if not continue_when_dead and our_dead:
                game_done = True
            else:
                # If we are here, it means either:
                # a) Our snake is alive (our_dead is False)
                # b) Our snake is dead BUT continue_when_dead is True

                if num_snakes > 1: # Multiplayer game
                    if len(alive_snakes) <= 1: # Game ends if one or zero snakes are left
                        game_done = True
                elif num_snakes == 1: # Single player game
                    if our_dead: # Game ends if our (the only) snake is dead
                        game_done = True
                    # If our snake is alive in a single player game, game_done remains False here,
                    # allowing the episode to continue.

            # pass actual done flag so winning reward and GAE work correctly
            memory.store_memory(obs_tensor.squeeze(0).cpu().numpy(), action, reward, game_done, log_prob, val)

            observation = next_obs
            obs_tensor = next_tensor
            json_obj = next_json
            episode_reward += reward
            episode_len += 1
            total_steps += 1

            if visualize and episode_num % visualize_interval == 0:  # <-- use visualize_interval here
                if screen and clock: # Ensure screen and clock are initialized
                    cont = visualize_step(env.get_json(), snake_colors, screen, clock, mute=mute, visualization_fps_value=visualization_fps)  # MODIFIED
                    if not cont:
                        visualize = False
                        quit_training_flag = True # Ensure training loop also exits
            
            if quit_training_flag: # Break from N_STEPS_COLLECT loop
                break

            if game_done:
                observation = game_state_to_observation(env.reset()[0])
                reset_sound_state_tracker()
                obs_tensor = torch.tensor(observation.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

                # --- Win detection for all snakes (moved to utils) ---
                win_flags, _ = determine_winner(info, num_snakes, our_snake_name)
                # lookup winner by flag in snake_names
                if 1 in win_flags:
                    winner_name = snake_names[win_flags.index(1)]
                else:
                    winner_name = "Draw"

                # Record wins for each snake
                for idx, win in enumerate(win_flags):
                    episode_wins_all[idx].append(win)
                    
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_len)
                episode_apples.append(info['snake_max_len'][0])  # record apples eaten for solo play

                episode_num += 1

                reason = info['snake_info'].get(0, 'Unknown')
                if reason != "Did not collide":
                    death_reasons.append(reason)

                # Calculate win percentage for agent (snake 0)
                win_percent = 100 * np.mean(episode_wins_all[0][-100:]) if episode_wins_all[0] else 0.0

                # Print death reason only in single player, win% and winner only if multiplayer
                if num_snakes > 1:
                    print(
                        f"Ep {episode_num:5d} | "
                        f"Step {total_steps:8d} | "
                        f"EpLen {episode_len:4d} | "
                        f"Reward {episode_reward:6.1f} | "
                        f"Win% {win_percent:6.2f} | "
                        f"Winner: {winner_name}"
                    )
                else:
                    print(
                        f"Ep {episode_num:5d} | "
                        f"Step {total_steps:8d} | "
                        f"EpLen {episode_len:4d} | "
                        f"Reward {episode_reward:6.1f} | "
                        f"Death reason: {reason}"
                    )

                if episode_num % plot_every == 0:
                    log_and_plot_training(
                        episode_rewards, 
                        episode_lengths, 
                        death_reasons,
                        episode_wins_all, 
                        episode_apples,
                        run_dir,
                        show_plot=False,
                        snake_names=snake_names
                    )

                if episode_num % save_interval == 0:
                    agent.save_model(os.path.join(run_dir, f"ppo_ep{episode_num}.pt"))

                episode_reward = 0
                episode_len = 0
                break  # stop this episode immediately (no bot‐only play)

        # Only update if we've collected a full batch
        if len(memory.rewards) >= N_STEPS_COLLECT:
            # if the last transition was terminal, set last_val_for_gae=0
            with torch.no_grad():
                _, _, last_val_for_gae = agent.choose_action(obs_tensor)
            agent.learn(memory, last_val_for_gae, global_step_counter=total_steps)
            memory.clear_memory()

    log_and_plot_training(
        episode_rewards, episode_lengths, death_reasons,
        episode_wins_all, episode_apples, run_dir,
        show_plot=False,
        snake_names=snake_names
    )

    # Final save
    agent.save_model(os.path.join(run_dir, "ppo_final.pt"))
    if visualize:
        close_visualization()
    print("Training complete.")
    return agent, episode_rewards, episode_lengths, run_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS, help="Total environment steps to train for")
    parser.add_argument("--map-size", type=int, nargs=2, default=[11, 11])
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--visualize-interval", type=int, default=1, help="How often to visualize the game (episodes)")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--opponents", type=str, nargs="+", default=[])
    parser.add_argument("--plot-every", type=int, default=100, help="How often to plot training stats (episodes)")
    parser.add_argument("--our-snake-name", type=str, default="Our snake", help="Name of our snake (agent)")
    parser.add_argument(
        "--continue-when-dead",
        action="store_true",
        help="If set, let episode continue after agent death; otherwise end on our snake death"
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="Mute all game sounds during training and visualization."
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Enable debug mode: pause each step for our snake and print debug info."
    )
    parser.add_argument(
        "--visualization-fps",
        type=int,
        default=40, # Default to fast visualization
        help="Frames per second for visualization. Set lower for slower visualization (e.g., 5)."
    )
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE}")
    train(
        total_timesteps=args.total_timesteps,  # Pass total_timesteps instead of episodes
        map_size=tuple(args.map_size),
        visualize=not args.no_visualize,
        save_interval=args.save_interval,
        visualize_interval=args.visualize_interval,  
        opponent_names=args.opponents,
        pretrained=args.pretrained,
        plot_every=args.plot_every,
        our_snake_name=args.our_snake_name,
        continue_when_dead=args.continue_when_dead,
        mute=args.mute,
        debug_mode=args.debug_mode,
        visualization_fps=args.visualization_fps,
    )
