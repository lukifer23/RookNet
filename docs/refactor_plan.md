# Refactor & Optimization Roadmap

## Milestone A – Codebase Consolidation
- [x] A1 Remove root-level duplicates (e.g. `alphazero_trainer.py`) keeping `src/` as single source of truth.
- [x] A2 Turn `src/` into an installable package (`pyproject.toml` / `setup.cfg`).
- [x] A3 Eliminate all `sys.path` hacks in favour of package imports.

## Milestone B – API Hardening
- [x] B1 Finalize `ChessEnvironment` API and add missing helpers (`select_move_with_temperature`, `get_result`, `board_to_tensor`).
- [x] B2 Merge duplicated MCTS implementations into one canonical module.
- [x] B3 Extend `StreamingReplayBuffer` to expose `add`, `extend`, `__len__`, `get_dataloader`, `total_games_added`, `sample`, `to_webdataset`.

## Milestone C – Move-Encoder Unification
- [x] C1 Lock policy vector to 4096-index flat mapping.
- [x] C2 Purge legacy 4672-mapping code.
- [x] C3 Write migration script to remap/pad existing checkpoints.

## Milestone D – Training Loop Repairs
- [x] D1 Compile model only once on GPU; guard with fallback to eager on MPS (`torch.compile` wrapped in try/except).
- [x] D2 Introduce dedicated GPU inference server; keep workers CPU-only.
   - [x] D2.1 Add `gpu_inference_server.py` module.
   - [x] D2.2 Trainer launches server and shares queues.
   - [x] D2.3 Replace worker local model with queue-based `RemoteModel` stub and wire queues.
   - [x] D2.4 Add timeout/retry, batch blocking, graceful shutdown, and Manager vs raw Queue toggle.
- [ ] D3 Persist and restore optimizer & scheduler state in checkpoints (pending upstream model refactor).

## Milestone E – Hyper-parameter Sanity
- [ ] E1 Scale CNN channels/blocks to ~64 M parameters.
- [ ] E2 Set realistic per-device batch sizes (CPU ↔ GPU).
- [ ] E3 Update `configs/config.v2.yaml` accordingly.

## Milestone F – Reliability & Observability
- [ ] F1 Reinstate and expand unit tests (encoder round-trip, MCTS visit accounting, replay I/O).
- [ ] F2 Add health checks & auto-respawn for worker processes.
- [ ] F3 Integrate TensorBoard/WandB logging gated by config.

## Milestone G – Packaging & Deployment
- [ ] G1 Create `pyproject.toml` with optional feature extras (GPU, webdataset, dev).
- [ ] G2 Add `make train`, `make eval`, `make web` convenience targets.
- [ ] G3 Set up CI workflow for linting, tests, and packaging.

## Milestone H – Performance Roadmap
- [ ] H1 Enable AMP + gradient accumulation for large effective batch.
- [ ] H2 Experiment with memory-efficient optimizers (PagedAdamW, Lion).
- [x] H3 Upgraded to PyTorch 2.8-dev; awaiting upstream fix for MPS Inductor dynamic shape bug.

---
Legend
- [ ] Pending
- [x] Done
- [-] N/A 