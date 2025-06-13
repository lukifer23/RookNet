# RookNet: Chess AI Master Plan

## Project Overview
RookNet is a superhuman chess AI leveraging a hybrid CNN+Transformer architecture, trained via AlphaZero-style dynamic self-play. The project is designed for scalability, reproducibility, and interactive analysis.

## Recent Changes
- **Project renamed to RookNet**
- Increased self-play and evaluation workers for faster training
- Enhanced logging and progress tracking
- Improved web GUI for live play and analysis
- Streamlined configuration and documentation

## Roadmap & Status
- **Phase 3: Dynamic Self-Play Training** (active)
  - 64M parameter model
  - 5 self-play workers, 5 evaluation workers, 5 data loading workers
  - Parallelized evaluation and detailed logging
  - Frequent checkpointing and progress saving
- **Web GUI**
  - Play against the latest model in real time
  - Visualize move suggestions and analysis

## Setup & Usage
- See `README.md` for setup and training instructions
- All essential scripts and configs are included
- Only the latest and best checkpoints are tracked in git

## Next Steps
- Continue long-term training on higher compute
- Monitor and analyze evaluation metrics
- Expand GUI features for deeper analysis

---

For technical details, see the code and configs. For questions, open an issue on GitHub.

