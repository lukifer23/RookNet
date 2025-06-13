import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import chess
import chess.engine
import chess.svg
import threading
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import subprocess
import sys
import os

# Import our model classes when ready
try:
    sys.path.append('src')
    from src.models.chess_transformer import ChessTransformer
    from src.utils.chess_env import ChessEnvironment
    import torch
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("⚠️  Model imports not available - will use engines only")

class ChessEngine:
    """Abstract base class for chess engines"""
    def __init__(self, name: str):
        self.name = name
        self.thinking = False
        
    def get_move(self, board: chess.Board, time_limit: float = 1.0) -> Optional[chess.Move]:
        """Get the best move for the current position"""
        raise NotImplementedError
        
    def get_evaluation(self, board: chess.Board) -> Optional[float]:
        """Get position evaluation in centipawns (positive = white advantage)"""
        raise NotImplementedError
        
    def cleanup(self):
        """Clean up engine resources"""
        pass

class UCIEngine(ChessEngine):
    """Base class for UCI engines like Stockfish and LC0"""
    def __init__(self, name: str, path: str, depth: int = None, nodes: int = None, time_limit: float = None):
        super().__init__(name)
        self.path = path
        self.depth = depth
        self.nodes = nodes
        self.time_limit = time_limit
        self.engine = None
        self.connect()
        
    def connect(self):
        try:
            # Increase timeout for resource-constrained systems
            self.engine = chess.engine.SimpleEngine.popen_uci(
                self.path, 
                timeout=30,  # 30 second connection timeout
                debug=False
            )
            print(f"✅ Connected to {self.name}")
        except Exception as e:
            print(f"❌ Failed to connect to {self.name}: {e}")
            self.engine = None
            
    def get_move(self, board: chess.Board, time_limit: float = 1.0) -> Optional[chess.Move]:
        if not self.engine:
            return None
            
        try:
            self.thinking = True
            
            # Create search limit based on engine configuration
            limit = chess.engine.Limit()
            if self.depth is not None:
                limit.depth = self.depth
            if self.nodes is not None:
                limit.nodes = self.nodes
            if self.time_limit is not None:
                limit.time = self.time_limit
            else:
                # Increase timeout for resource-constrained systems
                limit.time = max(time_limit * 3, 5.0)  # At least 5 seconds
                
            result = self.engine.play(board, limit)
            return result.move
        except Exception as e:
            print(f"{self.name} error: {e}")
            return None
        finally:
            self.thinking = False
            
    def get_evaluation(self, board: chess.Board) -> Optional[float]:
        if not self.engine:
            return None
            
        try:
            # Create analysis limit
            limit = chess.engine.Limit()
            if self.depth is not None:
                limit.depth = self.depth
            else:
                limit.depth = 10  # Default depth for analysis
                
            info = self.engine.analyse(board, limit)
            score = info["score"].relative
            if score.is_mate():
                return 10000 if score.mate() > 0 else -10000
            return float(score.score()) if score.score() is not None else 0.0
        except Exception as e:
            print(f"{self.name} evaluation error: {e}")
            return None
            
    def cleanup(self):
        if self.engine:
            try:
                self.engine.quit()
            except:
                pass

class StockfishEngine(UCIEngine):
    def __init__(self, path: str = "stockfish", depth: int = 1):
        super().__init__(f"Stockfish Depth {depth}", path, depth=depth)

class LC0Engine(UCIEngine):
    def __init__(self, path: str = "lc0", nodes: int = 100):
        super().__init__(f"LC0 ({nodes} nodes)", path, nodes=nodes)

class OurModelEngine(ChessEngine):
    def __init__(self, checkpoint_path: str):
        super().__init__(f"Our Model ({Path(checkpoint_path).name})")
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.env = None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.load_model()
        
    def load_model(self):
        if not MODEL_AVAILABLE:
            print("❌ Model not available")
            return
            
        try:
            # Load checkpoint with weights_only=False for our trusted checkpoints
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            config = checkpoint.get('config')
            
            # Initialize model with checkpoint config
            self.model = ChessTransformer(
                input_channels=config.input_channels,
                cnn_channels=config.cnn_channels,
                cnn_blocks=config.cnn_blocks,
                transformer_layers=config.transformer_layers,
                attention_heads=config.attention_heads
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.env = ChessEnvironment()
            print(f"✅ Loaded our model from {self.checkpoint_path}")
            
        except Exception as e:
            print(f"❌ Failed to load our model: {e}")
            self.model = None
            
    def get_move(self, board: chess.Board, time_limit: float = 1.0) -> Optional[chess.Move]:
        if not self.model:
            return None
            
        try:
            self.thinking = True
            
            # Convert board to tensor
            position_array = self.env.board_to_tensor(board)
            position_tensor = torch.from_numpy(position_array).float().unsqueeze(0).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                value, policy = self.model(position_tensor)
                policy_probs = torch.softmax(policy.squeeze(), dim=0).cpu().numpy()
            
            # Convert to legal moves with probabilities
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None
                
            move_probs = []
            for move in legal_moves:
                move_idx = move.from_square * 64 + move.to_square
                prob = policy_probs[move_idx] if move_idx < len(policy_probs) else 1e-8
                move_probs.append((move, prob))
            
            # Select move with highest probability
            best_move = max(move_probs, key=lambda x: x[1])[0]
            return best_move
            
        except Exception as e:
            print(f"Our model error: {e}")
            return None
        finally:
            self.thinking = False
            
    def get_evaluation(self, board: chess.Board) -> Optional[float]:
        if not self.model:
            return None
            
        try:
            position_array = self.env.board_to_tensor(board)
            position_tensor = torch.from_numpy(position_array).float().unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                value, _ = self.model(position_tensor)
                # Convert from [-1, 1] to centipawns
                return float(value.item()) * 1000
                
        except:
            return None

class HumanPlayer(ChessEngine):
    def __init__(self):
        super().__init__("Human Player")
        self.selected_move = None
        self.move_ready = False
        
    def set_move(self, move: chess.Move):
        self.selected_move = move
        self.move_ready = True
        
    def get_move(self, board: chess.Board, time_limit: float = 1.0) -> Optional[chess.Move]:
        self.thinking = True
        self.move_ready = False
        self.selected_move = None
        
        # Wait for human to select move
        while not self.move_ready:
            time.sleep(0.1)
            
        self.thinking = False
        return self.selected_move
        
    def get_evaluation(self, board: chess.Board) -> Optional[float]:
        return None  # Humans don't provide evaluations

class ChessBoard(tk.Canvas):
    def __init__(self, parent, size=400):
        super().__init__(parent, width=size, height=size, bg='white')
        self.size = size
        self.square_size = size // 8
        self.board = chess.Board()
        self.flipped = False
        self.selected_square = None
        self.legal_moves = []
        self.move_callback = None
        self.piece_images = {}
        
        self.bind("<Button-1>", self.on_click)
        self.load_pieces()  # Load pieces before drawing board
        self.draw_board()
        
    def load_pieces(self):
        """Load piece images (using text for now, can be replaced with actual images)"""
        self.piece_symbols = {
            chess.PAWN: ('♙', '♟'), chess.ROOK: ('♖', '♜'),
            chess.KNIGHT: ('♘', '♞'), chess.BISHOP: ('♗', '♝'),
            chess.QUEEN: ('♕', '♛'), chess.KING: ('♔', '♚')
        }
        
    def draw_board(self):
        self.delete("all")
        
        # Draw squares
        for rank in range(8):
            for file in range(8):
                x1 = file * self.square_size
                y1 = (7 - rank if not self.flipped else rank) * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                square = chess.square(file, rank)
                color = "#F0D9B5" if (rank + file) % 2 == 0 else "#B58863"
                
                # Highlight selected square
                if square == self.selected_square:
                    color = "#7FFFD4"
                    
                # Highlight legal move destinations
                if square in [move.to_square for move in self.legal_moves]:
                    color = "#90EE90"
                
                self.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                
        # Draw pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                self.draw_piece(square, piece)
                
        # Draw coordinates
        for i in range(8):
            # Files (a-h)
            file_char = chr(ord('a') + i)
            x = i * self.square_size + self.square_size // 2
            y = self.size - 10
            self.create_text(x, y, text=file_char, font=("Arial", 10))
            
            # Ranks (1-8)
            rank_char = str(8 - i if not self.flipped else i + 1)
            x = 10
            y = i * self.square_size + self.square_size // 2
            self.create_text(x, y, text=rank_char, font=("Arial", 10))
    
    def draw_piece(self, square: int, piece: chess.Piece):
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        x = file * self.square_size + self.square_size // 2
        y = (7 - rank if not self.flipped else rank) * self.square_size + self.square_size // 2
        
        symbol = self.piece_symbols[piece.piece_type][0 if piece.color else 1]
        self.create_text(x, y, text=symbol, font=("Arial", 24), fill="black")
    
    def on_click(self, event):
        if not self.move_callback:
            return
            
        file = event.x // self.square_size
        rank = 7 - (event.y // self.square_size) if not self.flipped else event.y // self.square_size
        
        if 0 <= file < 8 and 0 <= rank < 8:
            square = chess.square(file, rank)
            
            if self.selected_square is None:
                # First click - select piece
                if self.board.piece_at(square):
                    self.selected_square = square
                    self.legal_moves = [move for move in self.board.legal_moves 
                                      if move.from_square == square]
                    self.draw_board()
            else:
                # Second click - try to move
                move = chess.Move(self.selected_square, square)
                
                # Check for promotion
                piece = self.board.piece_at(self.selected_square)
                if (piece and piece.piece_type == chess.PAWN and 
                    ((piece.color and rank == 7) or (not piece.color and rank == 0))):
                    move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
                
                if move in self.board.legal_moves:
                    self.move_callback(move)
                    
                self.selected_square = None
                self.legal_moves = []
                self.draw_board()
    
    def set_board(self, board: chess.Board):
        self.board = board.copy()
        self.selected_square = None
        self.legal_moves = []
        self.draw_board()
        
    def flip_board(self):
        self.flipped = not self.flipped
        self.draw_board()

class ChessLabGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ChessLab - Advanced Chess Evaluation")
        self.root.geometry("1200x800")
        
        self.board = chess.Board()
        self.white_engine = None
        self.black_engine = None
        self.current_game_thread = None
        self.game_running = False
        self.game_paused = False
        
                                   # Available engines with ELO estimates
        self.available_engines = {
            "Human Player": lambda: HumanPlayer(),
            "Stockfish Depth 0 (~800 ELO)": lambda: StockfishEngine(depth=0),
            "Stockfish Depth 1 (~1200 ELO)": lambda: StockfishEngine(depth=1), 
            "Stockfish Depth 3 (~1800 ELO)": lambda: StockfishEngine(depth=3),
            "Stockfish Depth 5 (~2200 ELO)": lambda: StockfishEngine(depth=5),
            "Stockfish Depth 8 (~2600 ELO)": lambda: StockfishEngine(depth=8),
            "Stockfish Depth 12 (~3000 ELO)": lambda: StockfishEngine(depth=12),
            "Stockfish Depth 15 (~3200+ ELO)": lambda: StockfishEngine(depth=15),
            "LC0 100 nodes (~1500 ELO)": lambda: LC0Engine(nodes=100),
            "LC0 400 nodes (~2000 ELO)": lambda: LC0Engine(nodes=400),
            "LC0 800 nodes (~2400 ELO)": lambda: LC0Engine(nodes=800),
            "LC0 1600 nodes (~2800 ELO)": lambda: LC0Engine(nodes=1600),
        }
        
        # Add our model if available
        if MODEL_AVAILABLE:
            checkpoint_dir = Path("models/simple_small_checkpoints")
            if checkpoint_dir.exists():
                for checkpoint in sorted(checkpoint_dir.glob("*.pt")):
                    name = f"Our Model ({checkpoint.stem})"
                    self.available_engines[name] = lambda cp=str(checkpoint): OurModelEngine(cp)
        
        self.setup_ui()
        self.update_display()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Chess board
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        self.chess_board = ChessBoard(left_frame, size=500)
        self.chess_board.pack()
        self.chess_board.move_callback = self.human_move_made
        
        # Board controls
        board_controls = ttk.Frame(left_frame)
        board_controls.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(board_controls, text="Flip Board", 
                  command=self.chess_board.flip_board).pack(side=tk.LEFT)
        ttk.Button(board_controls, text="Reset Game", 
                  command=self.reset_game).pack(side=tk.LEFT, padx=(10, 0))
        
        # Right panel - Controls and info
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Engine selection
        engine_frame = ttk.LabelFrame(right_frame, text="Engine Configuration")
        engine_frame.pack(fill=tk.X, pady=(0, 10))
        
        # White player
        ttk.Label(engine_frame, text="White Player:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.white_var = tk.StringVar(value="Human Player")
        white_combo = ttk.Combobox(engine_frame, textvariable=self.white_var, 
                                  values=list(self.available_engines.keys()), width=25)
        white_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # Black player
        ttk.Label(engine_frame, text="Black Player:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.black_var = tk.StringVar(value="Stockfish Depth 1")
        black_combo = ttk.Combobox(engine_frame, textvariable=self.black_var,
                                  values=list(self.available_engines.keys()), width=25)
        black_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # Game controls
        game_frame = ttk.LabelFrame(right_frame, text="Game Controls")
        game_frame.pack(fill=tk.X, pady=(0, 10))
        
        controls_row1 = ttk.Frame(game_frame)
        controls_row1.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(controls_row1, text="Start Game", command=self.start_game)
        self.start_button.pack(side=tk.LEFT)
        
        self.pause_button = ttk.Button(controls_row1, text="Pause", command=self.pause_game, state="disabled")
        self.pause_button.pack(side=tk.LEFT, padx=(10, 0))
        
        self.stop_button = ttk.Button(controls_row1, text="Stop", command=self.stop_game, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # Time controls
        time_frame = ttk.Frame(game_frame)
        time_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(time_frame, text="Time per move (s):").pack(side=tk.LEFT)
        self.time_var = tk.StringVar(value="1.0")
        ttk.Entry(time_frame, textvariable=self.time_var, width=10).pack(side=tk.LEFT, padx=(10, 0))
        
        # Auto-play settings
        auto_frame = ttk.Frame(game_frame)
        auto_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.auto_play_var = tk.BooleanVar()
        ttk.Checkbutton(auto_frame, text="Auto-play mode", variable=self.auto_play_var).pack(side=tk.LEFT)
        
        ttk.Label(auto_frame, text="Games:").pack(side=tk.LEFT, padx=(20, 0))
        self.num_games_var = tk.StringVar(value="1")
        ttk.Entry(auto_frame, textvariable=self.num_games_var, width=5).pack(side=tk.LEFT, padx=(5, 0))
        
        # Game info
        info_frame = ttk.LabelFrame(right_frame, text="Game Information")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=8, width=50)
        info_scroll = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scroll.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Evaluation display
        eval_frame = ttk.LabelFrame(right_frame, text="Position Evaluation")
        eval_frame.pack(fill=tk.BOTH, expand=True)
        
        # Position eval bar
        eval_bar_frame = ttk.Frame(eval_frame)
        eval_bar_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(eval_bar_frame, text="Position:").pack(side=tk.LEFT)
        
        self.eval_canvas = tk.Canvas(eval_bar_frame, height=30, bg="gray90")
        self.eval_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Detailed evaluation
        self.eval_text = tk.Text(eval_frame, height=6, width=50)
        eval_text_scroll = ttk.Scrollbar(eval_frame, orient="vertical", command=self.eval_text.yview)
        self.eval_text.configure(yscrollcommand=eval_text_scroll.set)
        
        self.eval_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        eval_text_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
    def create_engines(self):
        """Create engine instances based on selection"""
        try:
            # Cleanup existing engines
            if self.white_engine:
                self.white_engine.cleanup()
                self.white_engine = None
            if self.black_engine:
                self.black_engine.cleanup()
                self.black_engine = None
                
            white_name = self.white_var.get()
            black_name = self.black_var.get()
            
            # Create engines with error checking
            try:
                self.white_engine = self.available_engines[white_name]()
                if hasattr(self.white_engine, 'engine') and self.white_engine.engine is None:
                    self.log_info(f"⚠️ Warning: {white_name} failed to connect")
            except Exception as e:
                self.log_info(f"❌ Failed to create white engine ({white_name}): {e}")
                return False
                
            try:
                # Create separate instance for black engine
                if white_name == black_name and not isinstance(self.white_engine, HumanPlayer):
                    # Different engines for same type to avoid conflicts
                    self.black_engine = self.available_engines[black_name]()
                else:
                    self.black_engine = self.available_engines[black_name]()
                    
                if hasattr(self.black_engine, 'engine') and self.black_engine.engine is None:
                    self.log_info(f"⚠️ Warning: {black_name} failed to connect")
            except Exception as e:
                self.log_info(f"❌ Failed to create black engine ({black_name}): {e}")
                return False
            
            self.log_info(f"✅ Created engines: {white_name} vs {black_name}")
            
        except Exception as e:
            messagebox.showerror("Engine Error", f"Failed to create engines: {e}")
            return False
        return True
        
    def start_game(self):
        if not self.create_engines():
            return
            
        self.game_running = True
        self.game_paused = False
        
        self.start_button.config(state="disabled")
        self.pause_button.config(state="normal")
        self.stop_button.config(state="normal")
        
        if self.auto_play_var.get():
            num_games = int(self.num_games_var.get())
            self.current_game_thread = threading.Thread(target=self.run_auto_games, args=(num_games,))
        else:
            self.current_game_thread = threading.Thread(target=self.run_single_game)
            
        self.current_game_thread.start()
        
    def pause_game(self):
        self.game_paused = not self.game_paused
        self.pause_button.config(text="Resume" if self.game_paused else "Pause")
        
    def stop_game(self):
        self.game_running = False
        self.game_paused = False
        
        self.start_button.config(state="normal")
        self.pause_button.config(state="disabled", text="Pause")
        self.stop_button.config(state="disabled")
        
    def reset_game(self):
        self.board = chess.Board()
        self.chess_board.set_board(self.board)
        self.update_display()
        
    def run_single_game(self):
        self.board = chess.Board()
        self.update_board_display()
        
        while not self.board.is_game_over() and self.game_running:
            while self.game_paused:
                time.sleep(0.1)
                
            if not self.game_running:
                break
                
            current_engine = self.white_engine if self.board.turn else self.black_engine
            
            # Get move from engine
            try:
                time_limit = float(self.time_var.get())
                
                # Check if engine is available
                if hasattr(current_engine, 'engine') and current_engine.engine is None:
                    self.log_info(f"Engine {current_engine.name} not available, stopping game")
                    break
                
                # Debug info
                self.log_info(f"Getting move from {current_engine.name} (Turn: {'White' if self.board.turn else 'Black'})")
                
                # Small pause to reduce CPU contention
                time.sleep(0.5)
                
                move = current_engine.get_move(self.board, time_limit)
                
                if move and move in self.board.legal_moves:
                    self.board.push(move)
                    color = "White" if not self.board.turn else "Black"
                    self.log_info(f"{color}: {move}")
                    self.update_board_display()
                else:
                    if move is None:
                        self.log_info(f"❌ Engine {current_engine.name} returned no move, attempting reconnect")
                        # Try to reconnect
                        if hasattr(current_engine, 'connect'):
                            self.log_info(f"🔄 Reconnecting {current_engine.name}")
                            current_engine.connect()
                            if current_engine.engine:
                                self.log_info(f"✅ Reconnected {current_engine.name}")
                                continue  # Try again
                    else:
                        self.log_info(f"❌ Invalid move {move} from {current_engine.name}, stopping game")
                    break
                    
            except Exception as e:
                self.log_info(f"Engine error ({current_engine.name}): {e}")
                break
                
        if self.game_running:
            result = self.board.result()
            self.log_info(f"Game over: {result}")
            
        self.stop_game()
        
    def run_auto_games(self, num_games):
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
        
        for game_num in range(1, num_games + 1):
            if not self.game_running:
                break
                
            self.log_info(f"\n=== Game {game_num}/{num_games} ===")
            self.board = chess.Board()
            self.update_board_display()
            
            while not self.board.is_game_over() and self.game_running:
                while self.game_paused:
                    time.sleep(0.1)
                    
                if not self.game_running:
                    break
                    
                current_engine = self.white_engine if self.board.turn else self.black_engine
                
                try:
                    time_limit = float(self.time_var.get())
                    
                    # Check if engine is available
                    if hasattr(current_engine, 'engine') and current_engine.engine is None:
                        self.log_info(f"Engine {current_engine.name} not available in game {game_num}")
                        break
                    
                    # Debug info
                    self.log_info(f"Getting move from {current_engine.name} (Turn: {'White' if self.board.turn else 'Black'})")
                    
                    # Small pause to reduce CPU contention
                    time.sleep(0.5)
                    
                    move = current_engine.get_move(self.board, time_limit)
                    
                    if move and move in self.board.legal_moves:
                        self.board.push(move)
                        self.log_info(f"{'White' if not self.board.turn else 'Black'}: {move}")
                        self.update_board_display()
                    else:
                        if move is None:
                            self.log_info(f"❌ No move from {current_engine.name} in game {game_num}")
                            # Try to reconnect
                            if hasattr(current_engine, 'connect'):
                                self.log_info(f"🔄 Attempting to reconnect {current_engine.name}")
                                current_engine.connect()
                                if current_engine.engine:
                                    self.log_info(f"✅ Reconnected {current_engine.name}")
                                    continue  # Try again
                        else:
                            self.log_info(f"❌ Invalid move {move} from {current_engine.name}")
                        break
                        
                except Exception as e:
                    self.log_info(f"Engine error in game {game_num} ({current_engine.name}): {e}")
                    break
                    
            if self.game_running and self.board.is_game_over():
                result = self.board.result()
                results[result] += 1
                self.log_info(f"Game {game_num} result: {result}")
                
        # Final statistics
        self.log_info(f"\n=== Final Results ===")
        self.log_info(f"White wins: {results['1-0']}")
        self.log_info(f"Black wins: {results['0-1']}")
        self.log_info(f"Draws: {results['1/2-1/2']}")
        
        self.stop_game()
        
    def human_move_made(self, move):
        """Called when human player makes a move via GUI"""
        if isinstance(self.white_engine, HumanPlayer) and self.board.turn == chess.WHITE:
            self.white_engine.set_move(move)
        elif isinstance(self.black_engine, HumanPlayer) and self.board.turn == chess.BLACK:
            self.black_engine.set_move(move)
            
    def update_board_display(self):
        """Update board display in main thread"""
        self.root.after(0, lambda: self.chess_board.set_board(self.board))
        self.root.after(0, self.update_display)
        
    def update_display(self):
        """Update all display elements"""
        self.update_evaluation()
        
    def update_evaluation(self):
        """Update position evaluation display"""
        try:
            # Get evaluations from both engines
            white_eval = None
            black_eval = None
            
            if self.white_engine and hasattr(self.white_engine, 'get_evaluation'):
                white_eval = self.white_engine.get_evaluation(self.board)
            if self.black_engine and hasattr(self.black_engine, 'get_evaluation'):
                black_eval = self.black_engine.get_evaluation(self.board)
                
            # Update evaluation bar (use Stockfish eval if available)
            eval_score = white_eval if white_eval is not None else (black_eval if black_eval is not None else 0)
            self.draw_eval_bar(eval_score)
            
            # Update evaluation text
            self.eval_text.delete(1.0, tk.END)
            
            if white_eval is not None:
                self.eval_text.insert(tk.END, f"White engine eval: {white_eval/100:.2f}\n")
            if black_eval is not None:
                self.eval_text.insert(tk.END, f"Black engine eval: {-black_eval/100:.2f}\n")
                
            # Material count
            material = self.count_material()
            self.eval_text.insert(tk.END, f"\nMaterial: White {material['white']}, Black {material['black']}\n")
            
            # Game status
            if self.board.is_check():
                self.eval_text.insert(tk.END, "CHECK!\n")
            if self.board.is_game_over():
                self.eval_text.insert(tk.END, f"Game Over: {self.board.result()}\n")
                
        except Exception as e:
            print(f"Evaluation update error: {e}")
            
    def draw_eval_bar(self, eval_score):
        """Draw evaluation bar"""
        self.eval_canvas.delete("all")
        width = self.eval_canvas.winfo_width()
        height = self.eval_canvas.winfo_height()
        
        if width <= 1:  # Canvas not ready
            return
            
        # Normalize score (-1000 to 1000 centipawns)
        normalized = max(-1000, min(1000, eval_score))
        ratio = (normalized + 1000) / 2000  # 0 to 1
        
        # Draw background
        self.eval_canvas.create_rectangle(0, 0, width, height, fill="lightgray", outline="")
        
        # Draw evaluation
        if ratio > 0.5:  # White advantage
            white_width = int((ratio - 0.5) * 2 * width)
            self.eval_canvas.create_rectangle(width//2, 0, width//2 + white_width, height, 
                                            fill="white", outline="black")
        else:  # Black advantage
            black_width = int((0.5 - ratio) * 2 * width)
            self.eval_canvas.create_rectangle(width//2 - black_width, 0, width//2, height,
                                            fill="black", outline="black")
            
        # Center line
        self.eval_canvas.create_line(width//2, 0, width//2, height, fill="red", width=2)
        
        # Score text
        score_text = f"{eval_score/100:.2f}"
        self.eval_canvas.create_text(width//2, height//2, text=score_text, fill="red", font=("Arial", 10, "bold"))
        
    def count_material(self):
        """Count material for both sides"""
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                       chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        
        white_material = sum(piece_values[piece.piece_type] 
                           for piece in self.board.piece_map().values() 
                           if piece.color == chess.WHITE)
        black_material = sum(piece_values[piece.piece_type] 
                           for piece in self.board.piece_map().values() 
                           if piece.color == chess.BLACK)
        
        return {"white": white_material, "black": black_material}
        
    def log_info(self, message):
        """Add message to info display"""
        self.root.after(0, lambda: self._log_info_safe(message))
        
    def _log_info_safe(self, message):
        """Thread-safe logging"""
        self.info_text.insert(tk.END, message + "\n")
        self.info_text.see(tk.END)
        
    def run(self):
        self.root.mainloop()
        
        # Cleanup
        if self.white_engine:
            self.white_engine.cleanup()
        if self.black_engine:
            self.black_engine.cleanup()

if __name__ == "__main__":
    app = ChessLabGUI()
    app.run()