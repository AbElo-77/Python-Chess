import torch, pandas
from torch.utils.data import Dataset, DataLoader
import chess


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

move_to_id, id_to_move = generate_moves_made(input_files); 

# ------------------------------ For Use By Algorithmic Interface 

def fen_to_tensor_cnn(fen: str) -> torch.Tensor: # For Both CNN and RNN As Of Now
    board = chess.Board(fen); 
    X = torch.zeros(12, 8, 8, dtype=torch.float32);
    for square, piece in board.piece_map().items():
        row_idx = 7 - (square // 8); 
        col_idx = square % 8; 
        X[pieces_as_indexes[piece.symbol()], row_idx, col_idx] = 1.0; 
    return X;

def fen_to_tensor_rnn(fen: str) -> torch.Tensor:
    board = chess.Board(fen); 
    X = torch.zeros(64, 12, dtype=torch.float32); 
    for square, piece in board.piece_map().items():
        X[square, pieces_as_indexes[piece.symbol()]] = 1.0; 
    return X; 


def fen_to_tensor_gnn(fen: str):
    board = chess.Board(fen); 

    node_features = torch.zeros(64, 12, dtype=torch.float32)

    for square, piece in board.piece_map().items():
            node_features[square, pieces_as_indexes[piece.symbol()]] = 1.0

    adjacency_matrix = torch.zeros(64, 64, dtype=torch.float32); 

    for square in range(64):
            rank, file = divmod(square, 8); 
            neighbors = [
                (rank + dr, file + dc)
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                if 0 <= rank + dr < 8 and 0 <= file + dc < 8
            ]
            for r, f in neighbors:
                adjacency_matrix[square, r*8+f] = 1.0; 
    
    node_features = node_features.unsqueeze(0); 
    adjacency_matrix = adjacency_matrix.unsqueeze(0); 

    Z = (node_features, adjacency_matrix); 
    return Z; 

# ----------------------------------------------------------------

class ChessData(Dataset):
    def __init__(self, input_files, move_to_id):
        self.dataframe = pandas.concat([pandas.read_csv(file) for file in input_files], ignore_index=True)
        self.move_to_id = move_to_id

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        board = chess.Board(row["chess_fen"]); 


        X = torch.zeros(12, 8, 8, dtype=torch.float32)
        for square, piece in board.piece_map().items():
            row_idx = 7 - (square // 8); 
            col_idx = square % 8; 
            X[pieces_as_indexes[piece.symbol()], row_idx, col_idx] = 1.0; 

        Y = torch.zeros(12, 8, 8, dtype=torch.float32)
        for square, piece in board.piece_map().items():
            row_idx = 7 - (square // 8); 
            col_idx = square % 8; 
            Y[pieces_as_indexes[piece.symbol()], row_idx, col_idx] = 1.0; 


        node_features = torch.zeros(64, 12, dtype=torch.float32)
        for square, piece in board.piece_map().items():
            node_features[square, pieces_as_indexes[piece.symbol()]] = 1.0

        adjacency_matrix = torch.zeros(64, 64, dtype=torch.float32)
        for square in range(64):
            rank, file = divmod(square, 8); 
            neighbors = [
                (rank + dr, file + dc)
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                if 0 <= rank + dr < 8 and 0 <= file + dc < 8
            ]
            for r, f in neighbors:
                adjacency_matrix[square, r*8+f] = 1.0; 

        Z = (node_features, adjacency_matrix); 

        y = torch.tensor(self.move_to_id[row["move_made"]], dtype=torch.long); 

        return X, Y, Z, y; 

def collate_function(batch):
    X_batch = torch.stack([item[0] for item in batch])
    Y_batch = torch.stack([item[1] for item in batch])
    Z_batch_node = [item[2][0] for item in batch]
    Z_batch_adj = [item[2][1] for item in batch]
    y_batch = torch.stack([item[3] for item in batch])
    return X_batch, Y_batch, list(zip(Z_batch_node, Z_batch_adj)), y_batch

def create_loader(batch_size, input_files, move_to_id):
    dataset = ChessData(input_files, move_to_id)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_function
    )
    return loader
