# filename: plot_training.py
"""
Reads training metrics from a CSV file and generates plots for all
available metrics, including optional round-based metrics if present.

Assumes the CSV file has at least columns:
episode,reward,avg_reward,agent_steps,total_steps,epsilon,avg_loss,duration_sec

Optionally plots these if available:
num_rounds,avg_round_reward
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

# --- Configuration ---
DEFAULT_CSV_PATH = "checkpoints/training_metrics.csv" # Default input CSV file name
DEFAULT_PLOTS_DIR = "plots" # Directory to save plots

def plot_metrics(csv_path, plots_dir):
    """
    Reads metrics from CSV and generates plots for available metrics.

    Args:
        csv_path (str): Path to the input CSV file.
        plots_dir (str): Directory where plots will be saved.
    """
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data from: {csv_path}")
        # Basic validation for core columns
        core_columns = ['episode', 'reward', 'avg_reward', 'agent_steps', 'total_steps', 'epsilon', 'avg_loss', 'duration_sec']
        if not all(col in df.columns for col in core_columns):
            missing = [col for col in core_columns if col not in df.columns]
            print(f"Error: CSV file is missing core required columns: {missing}")
            return
        if df.empty:
            print("Error: CSV file is empty.")
            return
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty or corrupted: {csv_path}")
        return
    except Exception as e:
        print(f"Error loading CSV file {csv_path}: {e}")
        return

    # Check for optional round metric columns
    has_round_metrics = 'num_rounds' in df.columns and 'avg_round_reward' in df.columns
    if has_round_metrics:
        print("Found round metrics (num_rounds, avg_round_reward) - will plot.")
    else:
        print("Round metrics (num_rounds, avg_round_reward) not found - skipping related plots.")


    # --- 2. Create Plots Directory ---
    if not os.path.exists(plots_dir):
        try:
            os.makedirs(plots_dir)
            print(f"Created directory: {plots_dir}")
        except OSError as e:
            print(f"Error creating directory {plots_dir}: {e}")
            return # Cannot proceed without plot directory

    # --- 3. Generate and Save Plots ---
    print("Generating plots...")

    # Define a rolling window size for potentially smoothing noisy plots if needed
    rolling_window = max(1, len(df) // 100) # Example: 1% of total episodes, min 1

    # Plot 1: Episode Rewards
    plt.figure(figsize=(12, 6))
    plt.plot(df['episode'], df['reward'], label='Episode Reward', alpha=0.6)
    plt.plot(df['episode'], df['avg_reward'], label='Avg Episode Reward (Rolling)', color='red', linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards per Episode")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(plots_dir, "episode_rewards_plot.png")
    try:
        plt.savefig(plot_path)
        print(f"Saved: {plot_path}")
    except Exception as e:
        print(f"Error saving plot {plot_path}: {e}")
    plt.close() # Close figure to free memory

    # Plot 2: Steps per Episode
    plt.figure(figsize=(12, 6))
    plt.plot(df['episode'], df['agent_steps'], label='Agent Steps per Episode', alpha=0.7)
    plt.plot(df['episode'], df['total_steps'], label='Total Steps per Episode', alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Number of Steps")
    plt.title("Steps per Training Episode")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(plots_dir, "steps_plot.png")
    try:
        plt.savefig(plot_path)
        print(f"Saved: {plot_path}")
    except Exception as e:
        print(f"Error saving plot {plot_path}: {e}")
    plt.close()

    # Plot 3: Epsilon (Exploration Rate)
    plt.figure(figsize=(12, 6))
    plt.plot(df['episode'], df['epsilon'], label='Epsilon')
    plt.xlabel("Episode")
    plt.ylabel("Epsilon Value")
    plt.title("Epsilon Decay Over Training Episodes")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(plots_dir, "epsilon_plot.png")
    try:
        plt.savefig(plot_path)
        print(f"Saved: {plot_path}")
    except Exception as e:
        print(f"Error saving plot {plot_path}: {e}")
    plt.close()

    # Plot 4: Average Loss
    plt.figure(figsize=(12, 6))
    plt.plot(df['episode'], df['avg_loss'], label='Average Loss (Reported)')
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.title("Average Training Loss per Episode")
    plt.legend()
    plt.grid(True)
    # Use log scale if loss values vary widely, but check for non-positive values first
    if (df['avg_loss'].dropna() > 0).all(): # Drop NA before checking positivity
         plt.yscale('log')
         plt.title("Average Training Loss per Episode (Log Scale)")
    else:
         print("Skipping log scale for loss due to zero or negative values.")

    plot_path = os.path.join(plots_dir, "loss_plot.png")
    try:
        plt.savefig(plot_path)
        print(f"Saved: {plot_path}")
    except Exception as e:
        print(f"Error saving plot {plot_path}: {e}")
    plt.close()

    # Plot 5: Duration per Episode
    plt.figure(figsize=(12, 6))
    plt.plot(df['episode'], df['duration_sec'], label='Episode Duration (seconds)', alpha=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Duration (seconds)")
    plt.title("Episode Duration Over Training")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(plots_dir, "duration_plot.png")
    try:
        plt.savefig(plot_path)
        print(f"Saved: {plot_path}")
    except Exception as e:
        print(f"Error saving plot {plot_path}: {e}")
    plt.close()

    # --- Optional Plots for Round Metrics ---
    if has_round_metrics:
        # Plot 6: Number of Rounds per Episode
        plt.figure(figsize=(12, 6))
        plt.plot(df['episode'], df['num_rounds'], label='Rounds per Episode', alpha=0.8)
        # Optional: Add rolling average
        # plt.plot(df['episode'], df['num_rounds'].rolling(window=rolling_window).mean(), label=f'Rounds per Episode ({rolling_window}-ep Roll Avg)', color='cyan', linestyle='--')
        plt.xlabel("Episode")
        plt.ylabel("Number of Rounds")
        plt.title("Number of Rounds Played per Episode")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(plots_dir, "num_rounds_plot.png")
        try:
            plt.savefig(plot_path)
            print(f"Saved: {plot_path}")
        except Exception as e:
            print(f"Error saving plot {plot_path}: {e}")
        plt.close()

        # Plot 7: Average Reward per Round
        plt.figure(figsize=(12, 6))
        plt.plot(df['episode'], df['avg_round_reward'], label='Avg Reward per Round (Episode)', alpha=0.8)
        # Optional: Add rolling average across episodes
        # plt.plot(df['episode'], df['avg_round_reward'].rolling(window=rolling_window).mean(), label=f'Avg Reward per Round ({rolling_window}-ep Roll Avg)', color='magenta', linestyle='--')
        plt.xlabel("Episode")
        plt.ylabel("Average Reward per Round")
        plt.title("Average Reward per Round within Each Episode")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(plots_dir, "avg_round_reward_plot.png")
        try:
            plt.savefig(plot_path)
            print(f"Saved: {plot_path}")
        except Exception as e:
            print(f"Error saving plot {plot_path}: {e}")
        plt.close()


    print("--- Plot generation finished ---")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Plot training metrics from a CSV file.")
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help=f"Path to the training metrics CSV file (default: {DEFAULT_CSV_PATH})"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_PLOTS_DIR,
        help=f"Directory to save the generated plots (default: {DEFAULT_PLOTS_DIR})"
    )

    # Parse arguments
    args = parser.parse_args()

    # Run the plotting function
    plot_metrics(args.csv, args.outdir)
