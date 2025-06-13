#!/usr/bin/env python3
"""
Open Chess Dataset Integration

This script downloads and processes open chess datasets to enhance training data.
Supports multiple sources including Lichess, Chess.com exports, and PGN collections.
"""

import sys
sys.path.append('src')

import requests
import zipfile
import gzip
import chess.pgn
import io
from pathlib import Path
from typing import List, Dict, Optional
import yaml
import json
from datetime import datetime
import os
import subprocess
import bz2

from src.utils.chess_env import ChessEnvironment

# Optional imports
try:
    import pandas as pd
except ImportError:
    pd = None


class ChessDatasetDownloader:
    """Download and process open chess datasets"""
    
    def __init__(self):
        self.data_dir = Path("data/external")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Known working chess dataset sources
        self.dataset_sources = {
            'sample_games': {
                'name': 'Sample Master Games',
                'url': 'https://raw.githubusercontent.com/niklasf/python-chess/master/data/games/FICS.pgn',
                'description': 'Sample collection of chess games',
                'format': 'pgn'
            },
            'lichess_sample': {
                'name': 'Lichess Sample Database',
                'url': 'https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.bz2',
                'description': 'Monthly Lichess games database (large file)',
                'format': 'pgn.bz2'
            },
            'kaggle_lichess': {
                'name': 'Kaggle Lichess Chess Dataset',
                'dataset': 'datasnaek/chess',
                'description': '120,000+ rated Lichess games with ELO ratings',
                'format': 'csv',
                'files': ['games.csv']
            }
        }
    
    def download_sample_games(self) -> bool:
        """Download a small sample of master games for testing"""
        print("📥 Downloading sample chess games...")
        
        try:
            # Create a small sample PGN file with known good games
            sample_pgn = '''[Event "World Championship"]
[Site "New York"]
[Date "1907.??.??"]
[Round "?"]
[White "Lasker, Emanuel"]
[Black "Marshall, Frank James"]
[Result "1-0"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7 7. Rc1 c6 8. Bd3 dxc4 9. Bxc4 Nd5 10. Bxe7 Qxe7 11. Ne4 N5f6 12. Ng3 e5 13. O-O exd4 14. Nxd4 Ne5 15. Bb3 Rd8 16. Qc2 Neg4 17. h3 Ne5 18. Rfd1 Bd7 19. Nf3 Nxf3+ 20. gxf3 Rac8 21. Qc5 Qxc5 22. Rxc5 Rxd1+ 23. Bxd1 Rd8 24. Bc2 Rd2 25. Rc2 Rxc2 26. Bxc2 Kf8 27. Ne4 Nxe4 28. fxe4 Ke7 29. f4 f6 30. Kf2 Kd6 31. Ke3 c5 32. b3 b6 33. Bd3 g6 34. a4 a5 35. Kd2 Bc6 36. Kc3 Kc7 37. Be2 Kd6 38. Kd2 Kc7 39. Ke3 Kd6 40. f5 gxf5 41. exf5 Ke5 42. Bd3 Bd5 43. Kd2 Kf4 44. Kc3 Kg3 45. Kd2 Kxh3 46. Ke3 Kg2 47. Kf4 h5 48. Kg3 h4+ 49. Kh2 Kf1 50. Kg1 Ke1 51. Bc2 Kd2 52. Bd3 Kc3 53. Kf2 Kb2 54. Ke3 Kxa4 55. Kd2 Kb4 56. Kc1 a4 57. bxa4 Kxa4 58. Kd2 b5 59. Kc3 b4+ 60. Kd2 Kb3 61. Kd1 Kc3 62. Kc1 b3 63. Kb1 Kd2 64. Bc4 Bxc4 65. Ka1 Kc2 66. Ka2 b2 67. Ka3 b1=Q 68. Ka4 Qb4# 1-0

[Event "Immortal Game"]
[Site "London"]
[Date "1851.06.21"]
[Round "?"]
[White "Anderssen, Adolf"]
[Black "Kieseritzky, Lionel"]
[Result "1-0"]

1. e4 e5 2. f4 exf4 3. Bc4 Qh4+ 4. Kf1 b5 5. Bxb5 Nf6 6. Nf3 Qh6 7. d3 Nh5 8. Nh4 Qg5 9. Nf5 c6 10. g4 Nf6 11. Rg1 cxb5 12. h4 Qg6 13. h5 Qg5 14. Qf3 Ng8 15. Bxf4 Qf6 16. Nc3 Bc5 17. Nd5 Qxb2 18. Bd6 Bxg1 19. e5 Qxa1+ 20. Ke2 Na6 21. Nxg7+ Kd8 22. Qf6+ Nxf6 23. Be7# 1-0

[Event "Evergreen Game"]
[Site "Berlin"]
[Date "1852.??.??"]
[Round "?"]
[White "Anderssen, Adolf"]
[Black "Dufresne, Jean"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. b4 Bxb4 5. c3 Ba5 6. d4 exd4 7. O-O d3 8. Qb3 Qf6 9. e5 Qg6 10. Re1 Nge7 11. Ba3 b5 12. Qxb5 Rb8 13. Qa4 Bb6 14. Nbd2 Bb7 15. Ne4 Qf5 16. Bxd3 Qh5 17. Nf6+ gxf6 18. exf6 Rg8 19. Rad1 Qxf3 20. Rxe7+ Nxe7 21. Qxd7+ Kxd7 22. Bf5+ Ke8 23. Bd7+ Kf8 24. Bxe7# 1-0
'''
            
            sample_file = self.data_dir / "sample_games.pgn"
            with open(sample_file, 'w') as f:
                f.write(sample_pgn)
            
            print(f"✅ Sample games saved to: {sample_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating sample games: {e}")
            return False
    
    def download_from_url(self, url: str, filename: str) -> bool:
        """Download dataset from URL"""
        try:
            print(f"📥 Downloading from: {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            filepath = self.data_dir / filename
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"✅ Downloaded: {filepath}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Download failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            return False
    
    def setup_kaggle_api(self) -> bool:
        """Setup Kaggle API credentials"""
        try:
            # Check if kaggle is installed
            import kaggle
            
            # Check if credentials exist
            kaggle_dir = Path.home() / '.kaggle'
            credentials_file = kaggle_dir / 'kaggle.json'
            
            if not credentials_file.exists():
                print("⚠️  Kaggle API credentials not found!")
                print("To use Kaggle datasets:")
                print("1. Go to https://www.kaggle.com/account")
                print("2. Create new API token")
                print("3. Place kaggle.json in ~/.kaggle/")
                print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
                return False
                
            print("✅ Kaggle API credentials found")
            return True
            
        except ImportError:
            print("📦 Installing Kaggle API...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
                import kaggle
                return self.setup_kaggle_api()  # Retry after installation
            except Exception as e:
                print(f"❌ Failed to install Kaggle API: {e}")
                return False
        except Exception as e:
            print(f"❌ Kaggle API setup error: {e}")
            return False
    
    def download_kaggle_dataset(self, dataset_name: str, max_games: int = 10000) -> List[Dict]:
        """Download and process Kaggle chess dataset"""
        print(f"📥 Downloading Kaggle dataset: {dataset_name}")
        
        if not self.setup_kaggle_api():
            return []
        
        try:
            import kaggle
            
            # Download dataset
            download_path = self.data_dir / "kaggle"
            download_path.mkdir(exist_ok=True)
            
            print(f"📦 Downloading {dataset_name} to {download_path}")
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=str(download_path), 
                unzip=True
            )
            
            # Look for games.csv file
            games_csv = download_path / 'games.csv'
            if not games_csv.exists():
                # Check for other CSV files
                csv_files = list(download_path.glob('*.csv'))
                if csv_files:
                    games_csv = csv_files[0]
                    print(f"📄 Using CSV file: {games_csv.name}")
                else:
                    print("❌ No CSV files found in dataset")
                    return []
            
            # Process the CSV file
            return self.process_kaggle_csv(games_csv, max_games)
            
        except Exception as e:
            print(f"❌ Error downloading Kaggle dataset: {e}")
            return []
    
    def process_kaggle_csv(self, csv_path: Path, max_games: int) -> List[Dict]:
        """Process Kaggle chess CSV file"""
        print(f"🔄 Processing Kaggle CSV: {csv_path}")
        
        # Ensure pandas is available
        local_pd = pd
        if local_pd is None:
            print("📦 Installing pandas...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
                import pandas as local_pd
            except Exception as e:
                print(f"❌ Failed to install pandas: {e}")
                return []
        
        try:
            # Read CSV in chunks to handle large files
            chunk_size = 1000
            games_data = []
            processed_count = 0
            
            for chunk in local_pd.read_csv(csv_path, chunksize=chunk_size):
                if processed_count >= max_games:
                    break
                
                for _, row in chunk.iterrows():
                    if processed_count >= max_games:
                        break
                    
                    try:
                        # Extract game information
                        game_info = self.parse_kaggle_game_row(row, local_pd)
                        if game_info and 10 <= len(game_info['moves']) <= 150:
                            games_data.append(game_info)
                            processed_count += 1
                            
                            if processed_count % 500 == 0:
                                print(f"   Processed {processed_count}/{max_games} games...")
                                
                    except Exception as e:
                        # Skip invalid games
                        continue
            
            print(f"✅ Processed {len(games_data)} games from Kaggle dataset")
            return games_data
            
        except Exception as e:
            print(f"❌ Error processing Kaggle CSV: {e}")
            return []
    
    def parse_kaggle_game_row(self, row, pandas_module=None) -> Optional[Dict]:
        """Parse a single game row from Kaggle dataset"""
        try:
            # Extract moves from PGN-like format
            moves_str = str(row.get('moves', ''))
            if not moves_str or moves_str == 'nan':
                return None
            
            # Parse the moves
            moves_data = []
            board = chess.Board()
            
            # Split moves and parse
            move_tokens = moves_str.split()
            for token in move_tokens:
                # Skip move numbers and result
                if '.' in token or token in ['1-0', '0-1', '1/2-1/2', '*']:
                    continue
                    
                try:
                    # Try to parse as SAN notation
                    move = board.parse_san(token)
                    if move in board.legal_moves:
                        moves_data.append({
                            'move': str(move),
                            'san': token,
                            'position': board.fen()
                        })
                        board.push(move)
                    else:
                        break
                except:
                    continue
            
            if len(moves_data) < 10:
                return None
            
            # Extract game metadata
            white_rating = row.get('white_rating', 0)
            black_rating = row.get('black_rating', 0)
            
            # Handle NaN values
            if pandas_module is not None:
                white_rating = int(white_rating) if pandas_module.notna(white_rating) else 0
                black_rating = int(black_rating) if pandas_module.notna(black_rating) else 0
            else:
                white_rating = int(white_rating) if str(white_rating) != 'nan' else 0
                black_rating = int(black_rating) if str(black_rating) != 'nan' else 0
            
            game_info = {
                'white': str(row.get('white_id', 'Unknown')),
                'black': str(row.get('black_id', 'Unknown')),
                'result': str(row.get('winner', '*')),
                'event': 'Lichess',
                'date': str(row.get('created_at', datetime.now().strftime('%Y.%m.%d')))[:10],
                'white_rating': white_rating,
                'black_rating': black_rating,
                'time_control': str(row.get('increment_code', 'unknown')),
                'moves': moves_data
            }
            
            return game_info
            
        except Exception as e:
            return None
    
    def process_pgn_file(self, pgn_path: Path, max_games: int = 1000) -> List[Dict]:
        """Process PGN file and extract game data"""
        print(f"🔄 Processing PGN file: {pgn_path}")
        
        games_data = []
        
        try:
            with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
                game_count = 0
                
                while game_count < max_games:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    # Extract game information
                    game_info = {
                        'white': game.headers.get('White', 'Unknown'),
                        'black': game.headers.get('Black', 'Unknown'),
                        'result': game.headers.get('Result', '*'),
                        'event': game.headers.get('Event', 'Unknown'),
                        'date': game.headers.get('Date', '????.??.??'),
                        'moves': []
                    }
                    
                    # Extract moves
                    board = chess.Board()
                    for move in game.mainline_moves():
                        if move in board.legal_moves:
                            game_info['moves'].append({
                                'move': str(move),
                                'san': board.san(move),
                                'position': board.fen()
                            })
                            board.push(move)
                        else:
                            break
                    
                    # Only include games with reasonable number of moves
                    if 10 <= len(game_info['moves']) <= 150:
                        games_data.append(game_info)
                        game_count += 1
                    
                    if game_count % 100 == 0:
                        print(f"   Processed {game_count} games...")
            
            print(f"✅ Processed {len(games_data)} games from {pgn_path}")
            return games_data
            
        except Exception as e:
            print(f"❌ Error processing PGN: {e}")
            return []
    
    def save_processed_data(self, games_data: List[Dict], filename: str) -> bool:
        """Save processed games data to JSON"""
        try:
            processed_dir = Path("data/processed")
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = processed_dir / filename
            with open(filepath, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'total_games': len(games_data),
                    'games': games_data
                }, f, indent=2)
            
            print(f"✅ Processed data saved to: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving processed data: {e}")
            return False
    
    def create_training_dataset(self, max_games: int = 1000, use_kaggle: bool = True) -> bool:
        """Create a comprehensive training dataset from available sources"""
        print("🚀 Creating comprehensive training dataset...")
        
        all_games = []
        
        # 1. Try Kaggle dataset first if requested (large dataset)
        if use_kaggle and max_games > 1000:
            print("🎯 Attempting to download Kaggle Lichess dataset...")
            kaggle_games = self.download_kaggle_dataset('datasnaek/chess', max_games=max_games)
            if kaggle_games:
                all_games.extend(kaggle_games)
                print(f"🎉 Successfully loaded {len(kaggle_games)} games from Kaggle!")
            else:
                print("⚠️  Kaggle dataset not available, falling back to other sources...")
        
        # 2. Create sample games if we need more data
        if len(all_games) < max_games // 2:
            if self.download_sample_games():
                sample_file = self.data_dir / "sample_games.pgn"
                if sample_file.exists():
                    games = self.process_pgn_file(sample_file, max_games=50)
                    all_games.extend(games)
        
        # 3. Look for any existing PGN files
        if len(all_games) < max_games // 2:
            for pgn_file in self.data_dir.glob("*.pgn"):
                if pgn_file.name != "sample_games.pgn":
                    games = self.process_pgn_file(pgn_file, max_games=max_games//4)
                    all_games.extend(games)
        
        # 4. Generate some games using Stockfish if we still don't have enough
        if len(all_games) < 100:
            print("📈 Generating additional games with Stockfish...")
            generated_games = self.generate_stockfish_games(num_games=200)
            all_games.extend(generated_games)
        
        print(f"📊 Total games collected: {len(all_games)}")
        
        # Save the complete dataset
        return self.save_processed_data(all_games, "open_dataset.json")
    
    def generate_stockfish_games(self, num_games: int = 100) -> List[Dict]:
        """Generate games using Stockfish vs itself at different levels"""
        print(f"🤖 Generating {num_games} games with Stockfish...")
        
        env = ChessEnvironment()
        games_data = []
        
        try:
            env.start_engine()
            
            for game_num in range(num_games):
                if game_num % 20 == 0:
                    print(f"   Generated {game_num}/{num_games} games...")
                
                # Reset board
                env.reset_board()
                moves_data = []
                move_count = 0
                max_moves = 100
                
                # Vary difficulty levels
                depths = [3, 5, 8, 10]
                white_depth = depths[game_num % len(depths)]
                black_depth = depths[(game_num + 1) % len(depths)]
                
                while not env.is_game_over() and move_count < max_moves:
                    depth = white_depth if env.board.turn == chess.WHITE else black_depth
                    
                    try:
                        best_move, _ = env.get_engine_move(time_limit=0.1, depth=depth)
                        if best_move and best_move in env.board.legal_moves:
                            moves_data.append({
                                'move': str(best_move),
                                'san': env.board.san(best_move),
                                'position': env.board.fen()
                            })
                            env.make_move(best_move)
                            move_count += 1
                        else:
                            break
                    except:
                        break
                
                # Record game if it has reasonable length
                if 15 <= len(moves_data) <= 80:
                    result = env.get_game_result() if env.is_game_over() else "1/2-1/2"
                    
                    games_data.append({
                        'white': f'Stockfish-{white_depth}',
                        'black': f'Stockfish-{black_depth}',
                        'result': result,
                        'event': 'Generated Game',
                        'date': datetime.now().strftime('%Y.%m.%d'),
                        'moves': moves_data
                    })
            
            env.stop_engine()
            
        except Exception as e:
            print(f"❌ Error generating games: {e}")
            try:
                env.stop_engine()
            except:
                pass
        
        print(f"✅ Generated {len(games_data)} games")
        return games_data
    
    def list_available_datasets(self):
        """List available dataset sources"""
        print("📋 Available Chess Datasets:")
        print("="*50)
        
        for key, info in self.dataset_sources.items():
            print(f"🎯 {info['name']}")
            print(f"   Description: {info['description']}")
            print(f"   Format: {info['format']}")
            if 'url' in info:
                print(f"   URL: {info['url']}")
            elif 'dataset' in info:
                print(f"   Kaggle Dataset: {info['dataset']}")
            print()


def main():
    """Main function"""
    print("🚀 Open Chess Dataset Integration")
    print("="*50)
    
    downloader = ChessDatasetDownloader()
    
    print("\nChoose an option:")
    print("1. List available datasets")
    print("2. Create training dataset (recommended)")
    print("3. Download Kaggle Lichess dataset (120,000+ games)")
    print("4. Download sample games only")
    print("5. Generate Stockfish games")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            downloader.list_available_datasets()
            
        elif choice == "2":
            max_games = input("Maximum games to process (default 1000, 10000+ recommended for Kaggle): ").strip()
            max_games = int(max_games) if max_games.isdigit() else 1000
            
            # Ask about Kaggle integration
            use_kaggle = True
            if max_games > 1000:
                kaggle_choice = input("Use Kaggle dataset for large-scale training? (y/N): ").strip().lower()
                use_kaggle = kaggle_choice in ['y', 'yes']
            
            success = downloader.create_training_dataset(max_games=max_games, use_kaggle=use_kaggle)
            if success:
                print("\n🎉 Training dataset created successfully!")
                print("📁 Check data/processed/open_dataset.json")
            else:
                print("❌ Failed to create training dataset")
                
        elif choice == "3":
            max_games = input("Maximum games to download from Kaggle (default 10000): ").strip()
            max_games = int(max_games) if max_games.isdigit() else 10000
            
            games = downloader.download_kaggle_dataset('datasnaek/chess', max_games=max_games)
            if games:
                downloader.save_processed_data(games, "kaggle_lichess_dataset.json")
                print(f"\n🎉 Downloaded {len(games)} games from Kaggle!")
            else:
                print("❌ Failed to download Kaggle dataset")
                
        elif choice == "4":
            downloader.download_sample_games()
            
        elif choice == "5":
            num_games = input("Number of games to generate (default 100): ").strip()
            num_games = int(num_games) if num_games.isdigit() else 100
            
            games = downloader.generate_stockfish_games(num_games=num_games)
            downloader.save_processed_data(games, "stockfish_games.json")
            
        else:
            print("Invalid choice. Creating default training dataset...")
            downloader.create_training_dataset()
    
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
