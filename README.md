# RookNet

Superhuman Chess AI using a hybrid CNN+Transformer AlphaZero-style architecture. This project implements scalable self-play training, evaluation, and a live web GUI for interactive play and analysis.

## Features
- 64M parameter CNN+Transformer model
- AlphaZero-style dynamic self-play with parallel workers
- Parallelized evaluation and detailed logging
- Live web GUI for playing against the model
- Cross-platform: macOS, Linux, Windows

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
   pip install -r requirements.txt
   ```
3. **Resume or start training:**
   ```sh
   python dynamic_self_play.py
   ```
4. **Launch the web GUI:**
   ```sh
   python chess_web_gui.py
   ```

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
