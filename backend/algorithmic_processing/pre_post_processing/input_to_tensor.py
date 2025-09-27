import pandas, torch
from torch.utils.data import Dataset, DataLoader
import chess, chess.pgn


# ------------------- Areas For Improvement 
# --------------------------------- Binary Check For Castling Rights
# --------------------------------- Move Turn With Row Of Bits

# ------------------- Converting FEN To  CNN, RNN, & GNN Tensors, Generating Moves Made From Dataset

pieces_as_indexes = {
    'P': 0, 'N': 1, 'B': 2, 'Q': 3, 'R': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'q': 9, 'r': 10, 'k': 11
}

input_files = ['./data/training_dataset/page_1.csv', 
               './data/training_dataset/page_2.csv', 
               './data/training_dataset/page_3.csv', 
               './data/training_dataset/page_4.csv', 
               './data/training_dataset/page_5.csv',
               './data/training_dataset/page_6.csv', 
               './data/training_dataset/page_7.csv', 
               './data/training_dataset/page_8.csv', 
               './data/training_dataset/page_9.csv', 
               './data/training_dataset/page_10.csv']

def generate_moves_made(input_files): 
    input_dataframe = pandas.concat(pandas.read_csv(file) for file in input_files); 

    unique_moves = sorted(input_dataframe["move_made"].unique()); 

    move_to_id = {m: i for i, m in enumerate(unique_moves)}; 
    id_to_move = {i: m for m, i in move_to_id.items()}; 

    return move_to_id, id_to_move; 


def fen_to_tensor_cnn(fen: str) -> torch.Tensor: 
    board = chess.Board(fen); 
    tensor = torch.zeros(12, 8, 8, dtype=torch.float32); 

    for square, piece in board.piece_map().items():
        row = 7 - (square // 8);  
        col = square % 8; 
        idx = pieces_as_indexes[piece.symbol()]; 
        tensor[idx, row, col] = 1.0; 
    
    return tensor

# ----------------------------------------------------------------------------------

def fen_to_tensor_rnn(fen: str) -> torch.Tensor: 
    board = chess.Board(fen); 
    tensor = torch.zeros((64, 12), dtype=torch.float32); 
    
    for square, piece in board.piece_map().items():
        idx = chess.square_rank(square) * 8 + chess.square_file(square); 
        tensor[idx, pieces_as_indexes[piece.symbol()]] = 1.0; 
    
    return tensor

# --- (Work In Progress) -------------------------------------------------------- 

def fen_to_tensor_gnn(fen:str) -> torch.Tensor: 
    board = chess.Board(fen); 
    tensor = torch.zeroes((64, 12), dtype=torch.float32); 
   
    for square, piece in board.piece_map().items(): 
        idx = square; 
        tensor[idx, pieces_as_indexes[piece.symbol()]] = 1.0; 

    adjacents = torch.zeros((64, 64), dtype=torch.float32); 

    for square in range(64):
        rank, file = divmod(square, 8); 
        neighbors = [
            (rank + dr, file + dc)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
            if 0 <= rank + dr < 8 and 0 <= file + dc < 8
        ]
        for r, f in neighbors:
            adjacents[square, r*8+f] = 1.0; 
    
    return tensor, adjacents
       

move_to_id, id_to_move = generate_moves_made(input_files); 

# ------------------- Dataset and DataLoader 

class ChessData(Dataset):
    def __init__(self, input_files, move_to_id): 
        self.dataframe = pandas.concat([pandas.read_csv(file) for file in input_files], ignore_index=True); 
        self.move_to_id = move_to_id; 

    def __len__(self):
        return len(self.dataframe);             

    def __getitem__(self, id):
        row = self.dataframe.iloc[id];          

        X = fen_to_tensor_cnn(row["chess_fen"]); 
        Y = fen_to_tensor_rnn(row["chess_fen"]); 
        Z = fen_to_tensor_gnn(row["chess_fen"]); 

        y = self.move_to_id[row["move_made"]]; 
        return X, Y, Z, y; 

all_dataset = ChessData(input_files, move_to_id); 

def create_loader(batch_size):

    batch_training = DataLoader(
    all_dataset, 
    batch_size = batch_size, 
    shuffle=True
    )  

    return batch_training