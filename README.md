# RookNet

 Chess AI using a hybrid CNN+Transformer AlphaZero-style architecture. This project implements scalable self-play training, evaluation, and a live web GUI for interactive play and analysis.

## Features
- 64M parameter CNN+Transformer model
- AlphaZero-style dynamic self-play with parallel workers
- Parallelized evaluation and detailed logging
- Live web GUI for playing against the model
- Runs on x86 CPUs with CUDA or AMD GPU acceleration

## Quick Start
1. **Clone the repo:**
   ```sh
   git clone https://github.com/lukifer23/RookNet.git
   cd RookNet
   ```
2. **Install dependencies:**
   ```sh
   python3 -m venv chess_ai_env
   source chess_ai_env/bin/activate
   pip install -e .[dev]
   ```
3. **Start or resume training:**
   ```sh
   python run.py train --config configs/config.v2.yaml
   ```
4. **Evaluate a model:**
   ```sh
   python run.py eval models/alpha_zero_checkpoints/latest_player.pt --games 5
   ```
5. **Launch the web GUI:**
   ```sh
   python run.py web
   ```

## Recommended Settings
- The default configuration trains a ~64M parameter model with 384 CNN channels and 20 residual blocks.
- Set `iterations` to 1000 and `games_per_iteration` to 50 for long runs.
- Evaluation happens every 5 iterations with 20 games to promote the best model.
- A GPU with at least **12 GB** of VRAM (e.g. RTX 3060) is recommended. CPU-only training is possible but very slow.

## Checkpoints
- `models/alpha_zero_checkpoints/latest_player.pt`: Most recent player model
- `models/alpha_zero_checkpoints/best_opponent.pt`: Best opponent model

## Documentation
- See `CHESS_AI_MASTER_PLAN.md` for project roadmap and technical details.
- See `configs/config.v2.yaml` for training configuration.

## Notes
- Data and logs are excluded from the repo for size and privacy.
- Only the latest and best checkpoints are included for reproducibility.

## License
MIT
