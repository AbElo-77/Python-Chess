import chess


# Testing Board Recognition and State; Updating Moves Properly and Checking Whether Moves/FENs Are Illegal 
# Tests Written On Designing API and Interface

def is_legal_move(fen: str, move: str):

    board = chess.Board(fen); 
    return board.is_legal(chess.Move.from_uci(move)); 
