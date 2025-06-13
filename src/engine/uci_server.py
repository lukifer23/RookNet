#!/usr/bin/env python3
"""
Native Chess Transformer – UCI Wrapper
-------------------------------------
This standalone script exposes the trained model as a UCI engine so it can plug
into common chess GUIs (Arena, CuteChess, etc.).

Supported subset of UCI:
• uci, isready, ucinewgame, position (startpos/FEN + moves), go (movetime|depth), quit

The engine is stateless between games except for the loaded neural net weights.
No multithreading or pondering implemented (kept minimal for stability).
"""
from __future__ import annotations

import sys
import sys
import chess
import json
import time
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parents[2] / 'src'))  # Add src to path

import torch
from models.chess_transformer import ChessTransformer
from utils.config_loader import load_config
from utils.move_encoder import get_policy_vector_size
from dynamic_self_play import MCTS

CONFIG = load_config("configs/config.v2.yaml")
POLICY_SIZE = get_policy_vector_size()

DEVICE = torch.device(CONFIG['system']['device'] if CONFIG['system']['device'] != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))

# Load model checkpoint (best opponent) -------------------------------------------------
ckpt_path = Path(CONFIG['training']['checkpoints']['dir']) / CONFIG['training']['checkpoints']['best_opponent_model']
if not ckpt_path.exists():
    print(f"info string No model checkpoint found at {ckpt_path}")
    sys.exit(1)

model_config = CONFIG['model']['chess_transformer']
model_config['policy_head_output_size'] = POLICY_SIZE

model = ChessTransformer(**model_config).to(DEVICE)
state = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(state.get('model_state_dict', state))
model.eval()

mcts = MCTS(model, DEVICE)

# ----------------------------------------------------------------------------
current_board = chess.Board()

def send(line: str):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def parse_position(tokens: List[str]):
    """Handle 'position' command."""
    global current_board
    if tokens[0] == 'startpos':
        current_board = chess.Board()
        moves = tokens[2:] if len(tokens) > 1 and tokens[1] == 'moves' else []
    elif tokens[0] == 'fen':
        fen = ' '.join(tokens[1:7])
        current_board = chess.Board(fen)
        moves = tokens[8:] if len(tokens) > 7 and tokens[7] == 'moves' else []
    else:
        return
    for mv in moves:
        try:
            current_board.push(chess.Move.from_uci(mv))
        except ValueError:
            continue


def best_move():
    policy, _ = mcts.search(current_board, add_noise=False)
    from dynamic_self_play import AlphaZeroTrainer
    move = AlphaZeroTrainer._select_move(policy, temperature=0, board=current_board)
    if move is None:
        return '0000'
    return move.uci()


# Main UCI loop -------------------------------------------------------------------------
while True:
    try:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if line == '':
            continue

        tokens = line.split()
        cmd = tokens[0]

        if cmd == 'uci':
            send('id name NativeChessTransformer')
            send('id author OpenAI')
            send('uciok')
        elif cmd == 'isready':
            send('readyok')
        elif cmd == 'ucinewgame':
            current_board = chess.Board()
        elif cmd == 'position':
            parse_position(tokens[1:])
        elif cmd == 'go':
            # Simple handling: respect movetime X or depth Y
            movetime_ms = None
            depth = None
            if 'movetime' in tokens:
                movetime_ms = int(tokens[tokens.index('movetime') + 1])
            if 'depth' in tokens:
                depth = int(tokens[tokens.index('depth') + 1])

            start = time.time()
            best = best_move()
            # If time limit specified, ensure we wait roughly that long to mimic engine thinking
            if movetime_ms is not None:
                elapsed = (time.time() - start) * 1000
                if elapsed < movetime_ms:
                    time.sleep(max(0, (movetime_ms - elapsed) / 1000))
            send(f'bestmove {best}')
        elif cmd == 'quit':
            break
    except Exception as e:
        send(f'info string error {e}')
        break 