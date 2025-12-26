import argparse
import sys
import os

# PyTorch components
import torch

# Import logic for pipeline stages
from data_preprocessing import run_preprocessing
from train import run_training
from rl_train import run_rl
from generate import run_generation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VEGFR2 HeteroEncoder Pipeline")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['preprocess', 'train', 'rl', 'generate', 'all'],
                        help='Pipeline execution mode')

    # Arguments for molecule generation
    parser.add_argument('--energy', type=float, default=-10.0,
                        help='Target binding energy for generation')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of molecules to generate')
    parser.add_argument('--threshold', type=float, default=-10,
                        help='Molecular selection threshold')
    parser.add_argument('--noise', type=float, default=0.2,
                        help='Noise for Fine-tuning and generation')

    args = parser.parse_args()

    # Data Preprocessing stage
    if args.mode == 'preprocess' or args.mode == 'all':
        print("\n--- Running Preprocessing ---")
        run_preprocessing()

    # Supervised Learning stage
    if args.mode == 'train' or args.mode == 'all':
        print("\n--- Running Supervised Training ---")
        run_training(epochs=30)

    # Reinforcement Learning stage
    if args.mode == 'rl' or args.mode == 'all':
        print("\n--- Running RL Fine-tuning ---")
        run_rl(epochs=10, )

    # Generation stage
    if args.mode == 'generate':
        print("\n--- Running Molecule Generation ---")
        run_generation(total_attempts=args.samples,
                       target_energy=args.energy,
                       seed_energy_threshold=args.threshold,
                       noise=args.noise)