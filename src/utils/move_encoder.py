# src/utils/move_encoder.py
import chess

# Define a fixed, canonical mapping of moves.
# This is a simplified version of the AlphaZero-style 8x8x73 representation.
# We will use a flat vector of size 4672 for all possible moves.
# This ensures no collisions and provides a stable target for the policy head.

# Generate all possible queen moves (from, to)
QUEEN_MOVES = []
for from_sq in chess.SQUARES:
    for to_sq in chess.SQUARES:
        if from_sq == to_sq:
            continue
        # Check if it's a valid queen-like move (rank, file, or diagonal)
        if chess.square_distance(from_sq, to_sq) == 1 or \
           abs(chess.square_file(from_sq) - chess.square_file(to_sq)) == abs(chess.square_rank(from_sq) - chess.square_rank(to_sq)) or \
           chess.square_file(from_sq) == chess.square_file(to_sq) or \
           chess.square_rank(from_sq) == chess.square_rank(to_sq):
            QUEEN_MOVES.append(chess.Move(from_sq, to_sq))

# Generate all possible knight moves
KNIGHT_MOVES = []
for from_sq in chess.SQUARES:
    for to_sq in chess.SQUARES:
        if chess.square_distance(from_sq, to_sq) > 2:
            continue
        file_dist = abs(chess.square_file(from_sq) - chess.square_file(to_sq))
        rank_dist = abs(chess.square_rank(from_sq) - chess.square_rank(to_sq))
        if file_dist == 1 and rank_dist == 2:
            KNIGHT_MOVES.append(chess.Move(from_sq, to_sq))
        if file_dist == 2 and rank_dist == 1:
            KNIGHT_MOVES.append(chess.Move(from_sq, to_sq))

# Generate underpromotions (knight, bishop, rook)
UNDERPROMOTIONS = []
for from_file in range(8):
    for to_file in range(8):
        # Pawn on 7th rank promoting
        if abs(from_file - to_file) <= 1:
            from_sq_w = chess.square(from_file, 6)
            to_sq_w = chess.square(to_file, 7)
            for piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
                UNDERPROMOTIONS.append(chess.Move(from_sq_w, to_sq_w, promotion=piece))
        # Pawn on 2nd rank promoting
        if abs(from_file - to_file) <= 1:
            from_sq_b = chess.square(from_file, 1)
            to_sq_b = chess.square(to_file, 0)
            for piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
                UNDERPROMOTIONS.append(chess.Move(from_sq_b, to_sq_b, promotion=piece))

# Create the full move mapping
# This list is ordered and provides the canonical index for each move.
CANONICAL_MOVE_LIST = sorted(list(set(QUEEN_MOVES + KNIGHT_MOVES + UNDERPROMOTIONS)), key=lambda m: m.uci())

# Create a dictionary for fast lookups
MOVE_TO_INDEX = {move: i for i, move in enumerate(CANONICAL_MOVE_LIST)}
INDEX_TO_MOVE = {i: move for i, move in enumerate(CANONICAL_MOVE_LIST)}

POLICY_VECTOR_SIZE = len(CANONICAL_MOVE_LIST)

def encode_move(move: chess.Move) -> int:
    """
    Converts a chess.Move object to its canonical integer index.
    Handles queen promotions by mapping them to their non-promoting equivalent.
    """
    # Handle standard queen promotions by mapping them to the non-promoting move
    if move.promotion == chess.QUEEN:
        move.promotion = None
    
    return MOVE_TO_INDEX.get(move)

def decode_move(index: int) -> chess.Move:
    """
    Converts an integer index back to its canonical chess.Move object.
    """
    return INDEX_TO_MOVE.get(index)

def get_policy_vector_size() -> int:
    """
    Returns the total size of the policy vector.
    """
    return POLICY_VECTOR_SIZE

def get_legal_move_mask(board: chess.Board) -> list[int]:
    """
    Returns a list of indices corresponding to the legal moves in the current position.
    """
    legal_move_indices = []
    for move in board.legal_moves:
        idx = encode_move(move)
        if idx is not None:
            legal_move_indices.append(idx)
    return legal_move_indices 