#!/usr/bin/env python3
"""
Aves Horizons Training Script

Clean, modular training system using StableBaselines3 PPO with self-play.
"""

import json
import argparse
from pathlib import Path
from dataclasses import asdict

from training import TrainingConfig, AvesHorizonsTrainer, create_trainer_from_config_file
from utils.entry_points import entry_point_manager, handle_common_errors, print_config
from utils.config import ModelConfig


@handle_common_errors
def main():
    """Main training entry point."""
    parser = entry_point_manager.create_base_parser(
        'Aves Horizons Training System - StableBaselines3 PPO with Self-Play'
    )
    entry_point_manager.add_training_args(parser)
    
    # Add training-specific arguments
    parser.add_argument('--total-timesteps', type=int, default=1000000, 
                       help='Total training timesteps')
    parser.add_argument('--config-file', type=Path, 
                       help='Load configuration from JSON file')
    parser.add_argument('--save-config', type=Path,
                       help='Save current configuration to JSON file')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate existing model, do not train')
    parser.add_argument('--load-model', type=Path,
                       help='Load existing model for training/evaluation')
    
    # PPO-specific arguments
    ppo_group = parser.add_argument_group('PPO Options')
    ppo_group.add_argument('--n-steps', type=int, default=2048,
                          help='Steps per environment per update')
    ppo_group.add_argument('--gamma', type=float, default=0.99,
                          help='Discount factor')
    ppo_group.add_argument('--gae-lambda', type=float, default=0.95,
                          help='GAE lambda parameter')
    ppo_group.add_argument('--clip-range', type=float, default=0.2,
                          help='PPO clipping parameter')
    ppo_group.add_argument('--ent-coef', type=float, default=0.01,
                          help='Entropy coefficient')
    ppo_group.add_argument('--vf-coef', type=float, default=0.5,
                          help='Value function coefficient')
    
    # Self-play arguments
    selfplay_group = parser.add_argument_group('Self-Play Options')
    selfplay_group.add_argument('--selfplay-freq', type=int, default=50000,
                               help='Timesteps between self-play model updates')
    selfplay_group.add_argument('--max-pool-size', type=int, default=10,
                               help='Maximum opponent pool size')
    selfplay_group.add_argument('--opponent-probs', nargs=3, type=float, 
                               default=[0.1, 0.2, 0.7], metavar=('RANDOM', 'HEURISTIC', 'SELFPLAY'),
                               help='Opponent selection probabilities')
    
    args = parser.parse_args()
    
    # Create or load configuration
    if args.config_file and args.config_file.exists():
        print(f"Loading configuration from: {args.config_file}")
        trainer = create_trainer_from_config_file(args.config_file)
        config = trainer.config
    else:
        # Create configuration from command line arguments
        config = TrainingConfig(
            # Model parameters - use new ShipNN parameters if provided, otherwise use legacy
            hidden_dim=args.hidden_dim if hasattr(args, 'hidden_dim') else (args.d_model or 128),
            encoder_layers=args.encoder_layers if hasattr(args, 'encoder_layers') else 2,
            transformer_layers=args.transformer_layers if hasattr(args, 'transformer_layers') else (args.num_layers or 3),
            decoder_layers=args.decoder_layers if hasattr(args, 'decoder_layers') else 2,
            n_heads=args.n_heads if hasattr(args, 'n_heads') else (args.n_head or 4),
            dim_feedforward=args.dim_feedforward if hasattr(args, 'dim_feedforward') else 256,
            
            # Game parameters
            n_teams=ModelConfig.DEFAULT_N_TEAMS,
            ships_per_team=args.controlled_team_size // ModelConfig.DEFAULT_N_TEAMS,
            sequence_length=args.sequence_length,
            world_size=tuple(args.world_size),
            
            # PPO parameters
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            
            # Self-play parameters
            selfplay_update_freq=args.selfplay_freq,
            max_opponent_pool_size=args.max_pool_size,
            opponent_selection_probs={
                "random": args.opponent_probs[0],
                "heuristic": args.opponent_probs[1], 
                "selfplay": args.opponent_probs[2]
            },
            
            # Training control
            total_timesteps=args.total_timesteps,
            eval_freq=args.eval_frequency,
            save_freq=args.save_frequency
        )
        
        trainer = AvesHorizonsTrainer(config)
    
    # Save configuration if requested
    if args.save_config:
        config_dict = asdict(config)
        # Convert Path objects to strings for JSON serialization
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        with open(args.save_config, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to: {args.save_config}")
    
    # Load existing model if specified
    if args.load_model:
        if not args.load_model.exists():
            print(f"Error: Model file not found: {args.load_model}")
            return
        trainer.load_model(args.load_model)
        print(f"Loaded existing model: {args.load_model}")
    
    # Display configuration
    print_config(asdict(config), "Training Configuration")
    
    # Either evaluate or train
    if args.eval_only:
        if not args.load_model:
            print("Error: --eval-only requires --load-model")
            return
        
        print("Running evaluation...")
        results = trainer.evaluate()
        print(f"Evaluation results: {results}")
        
    else:
        # Display training summary
        print(f"\n🚀 Starting Aves Horizons Training")
        print(f"📊 Environment: Deathmatch Self-Play")
        print(f"🤖 Model: ShipNN ({config.hidden_dim}d, E{config.encoder_layers}/T{config.transformer_layers}/D{config.decoder_layers}, {config.n_heads}H)")
        print(f"🎯 Algorithm: PPO")
        print(f"⚡ Total timesteps: {config.total_timesteps:,}")
        print(f"🔄 Self-play update frequency: {config.selfplay_update_freq:,}")
        print(f"💾 Models will be saved to: {config.model_dir}")
        print(f"📈 Logs will be saved to: {config.log_dir}")
        print(f"📊 Tensorboard logs: {config.tensorboard_log}")
        
        # Start training
        try:
            trainer.train(config.total_timesteps)
            
            # Final evaluation
            print("\n🎯 Running final evaluation...")
            results = trainer.evaluate(n_eval_episodes=20)
            print(f"Final evaluation results: {results}")
            
        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            raise
        finally:
            trainer.close()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()