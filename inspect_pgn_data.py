#!/usr/bin/env python3
"""
Quick inspection of PGN data to understand why conversion failed
"""

import chess.pgn
from pathlib import Path

def inspect_lichess_pgn():
    """Inspect the first few games in the Lichess database"""
    
    pgn_path = "lichess_db_standard_rated_2013-01.pgn"
    
    if not Path(pgn_path).exists():
        print("❌ PGN file not found!")
        return
    
    print("🔍 INSPECTING LICHESS PGN FORMAT")
    print("=" * 50)
    
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
        for game_num in range(5):  # Check first 5 games
            try:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                print(f"\n🎮 GAME {game_num + 1}:")
                print(f"   📊 Headers: {dict(game.headers)}")
                
                # Check ELO ratings
                white_elo = game.headers.get('WhiteElo', 'Unknown')
                black_elo = game.headers.get('BlackElo', 'Unknown')
                print(f"   🏆 White ELO: {white_elo}")
                print(f"   🏆 Black ELO: {black_elo}")
                
                # Check game moves
                moves = []
                for node in game.mainline():
                    moves.append(str(node.move))
                    if len(moves) >= 10:  # First 10 moves
                        break
                
                print(f"   🎯 First moves: {' '.join(moves[:10])}")
                print(f"   📈 Total moves in game: {len(list(game.mainline()))}")
                
            except Exception as e:
                print(f"   ❌ Error reading game {game_num + 1}: {e}")
                continue

if __name__ == "__main__":
    inspect_lichess_pgn() 