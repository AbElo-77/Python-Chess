import torch, pandas
from torch.utils.data import Dataset, DataLoader
import chess


pieces_as_indexes = {
    'P': 0, 'N': 1, 'B': 2, 'Q': 3, 'R': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'q': 9, 'r': 10, 'k': 11
}

input_files = ['./data/training_dataset/all_files.csv']

def generate_moves_made(input_files): 
    input_dataframe = pandas.concat(pandas.read_csv(file) for file in input_files) 

    unique_moves = sorted(input_dataframe["move_made"].unique()) 

    move_to_id = {m: i for i, m in enumerate(unique_moves)} 
    id_to_move = {i: m for m, i in move_to_id.items()} 

    return move_to_id, id_to_move 

# ------------------------------ For Use By Algorithmic Interface 

def fen_to_tensor_cnn(fen: str) -> torch.Tensor: 
    board = chess.Board(fen) 
    X = torch.zeros(12, 8, 8, dtype=torch.float32) 
    for square, piece in board.piece_map().items():
        row_idx = 7 - (square // 8) 
        col_idx = square % 8 
        X[pieces_as_indexes[piece.symbol()], row_idx, col_idx] = 1.0 
    return X

def fen_to_tensor_rnn(fen: str) -> torch.Tensor:
    board = chess.Board(fen)
    X = torch.zeros(8, 8, 12, dtype=torch.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)  
        col = square % 8
        X[row, col, pieces_as_indexes[piece.symbol()]] = 1.0
    return X  


def fen_to_tensor_gnn(fen: str):
    board = chess.Board(fen) 

    node_features = torch.zeros(64, 12, dtype=torch.float32)
    adjacency_matrix = torch.zeros(64, 64, dtype=torch.float32) 

    for square, piece in board.piece_map().items():
            node_features[square, pieces_as_indexes[piece.symbol()]] = 1.0

            for move in list(board.legal_moves):

                if move.from_square == square: 
                    adjacency_matrix[square, move.to_square] = 1.0

    # for square in range(64):
            # rank, file = divmod(square, 8)                   ---- Encodes Adjacency Spatially
            # neighbors = [                                     ----- Replaced with Legal Move Encoding
            #     (rank + dr, file + dc)
            #     for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
            #     if 0 <= rank + dr < 8 and 0 <= file + dc < 8
            # ]
            # for r, f in neighbors:
            #     adjacency_matrix[square, r*8+f] = 1.0 
    
    node_features = node_features.unsqueeze(0) 
    adjacency_matrix = adjacency_matrix.unsqueeze(0) 

    Z = (node_features, adjacency_matrix) 
    return Z 

# ----------------------------------------------------------------

class ChessData(Dataset):
    def __init__(self, input_files, move_to_id):

        dfs = [] 
        for file in input_files:
            try:
                df = pandas.read_csv(file) 

                df['move_made'] = df['move_made'].astype(str).str.strip() 
                dfs.append(df) 

            except Exception as e:
                print(f"Error IN {file}: {e}") 
                continue 
            
        self.dataframe = pandas.concat(dfs, ignore_index=True) 
        self.move_to_id = move_to_id 

    def __len__(self):
        return len(self.dataframe) 

    def __getitem__(self, idx):
        try:

            row = self.dataframe.iloc[idx] 

            chess_fen = row.chess_fen 
            move_made = str(row.move_made).strip() 
            
            board = chess.Board(chess_fen) 
            
            X = torch.zeros(12, 8, 8, dtype=torch.float32) 
            for square, piece in board.piece_map().items():
                row_idx = 7 - (square // 8) 
                col_idx = square % 8 
                X[pieces_as_indexes[piece.symbol()], row_idx, col_idx] = 1.0 
                
        except Exception as e:
            print(f"Error processing item {idx}: {e}") 
            raise 

        Y = torch.zeros(8, 8, 12, dtype=torch.float32)
        for square, piece in board.piece_map().items():
            r = 7 - (square // 8)  
            c = square % 8
            Y[r, c, pieces_as_indexes[piece.symbol()]] = 1.0

        Y = Y.permute(2, 0, 1) 

        node_features = torch.zeros(64, 12, dtype=torch.float32) 
        node_features = torch.zeros(64, 12, dtype=torch.float32)
        adjacency_matrix = torch.zeros(64, 64, dtype=torch.float32) 

        for square, piece in board.piece_map().items():
            node_features[square, pieces_as_indexes[piece.symbol()]] = 1.0

            for move in list(board.legal_moves):

                if move.from_square == square: 
                    adjacency_matrix[square, move.to_square] = 1.0

        # for square in range(64):
            # rank, file = divmod(square, 8)                   ---- Encodes Adjacency Spatially
            # neighbors = [                                     ----- Replaced with Legal Move Encoding
            #     (rank + dr, file + dc)
            #     for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
            #     if 0 <= rank + dr < 8 and 0 <= file + dc < 8
            # ]
            # for r, f in neighbors:
            #     adjacency_matrix[square, r*8+f] = 1.0 
    
        node_features = node_features.unsqueeze(0) 
        adjacency_matrix = adjacency_matrix.unsqueeze(0) 

        Z = (node_features, adjacency_matrix) 

        try:
            y = torch.tensor(self.move_to_id[move_made], dtype=torch.long)
        except KeyError:
            raise KeyError(f"Move '{move_made}' not found in move dictionary") 

        return X, Y, Z, y

def collate_function(batch):
    X_batch = torch.stack([item[0] for item in batch]) 
    Y_batch = torch.stack([item[1] for item in batch])  
    

    Z_batch_node = torch.stack([item[2][0] for item in batch])  
    Z_batch_adj = torch.stack([item[2][1] for item in batch])  
    Z_batch = (Z_batch_node, Z_batch_adj) 
    
    y_batch = torch.tensor([item[3] for item in batch], dtype=torch.long) 
    return X_batch, Y_batch, Z_batch, y_batch 

def create_loader(batch_size, input_files, move_to_id):
    dataset = ChessData(input_files, move_to_id) 
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_function
    )

    class SafeDataLoader:
        def __init__(self, loader):
            self._loader = loader
            dataset_len = len(self._loader.dataset) if hasattr(self._loader, 'dataset') else 0
            bs = self._loader.batch_size if getattr(self._loader, 'batch_size', None) else batch_size

            if bs is None or bs <= 0:
                bs = 1
            self._max_batches = (dataset_len + bs - 1) // bs if dataset_len > 0 else 0

        def __iter__(self):
            self._iter = iter(self._loader)
            self._count = 0
            return self

        def __next__(self):
            if self._count >= self._max_batches:
                raise StopIteration
            batch = next(self._iter)
            self._count += 1
            return batch

        def __len__(self):
            return self._max_batches
        
        def __getattr__(self, name):
            return getattr(self._loader, name)

    return SafeDataLoader(loader)