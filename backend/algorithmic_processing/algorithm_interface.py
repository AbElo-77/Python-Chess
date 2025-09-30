import torch
from backend.algorithmic_processing.models.neural_network_models.convolutional_nn_model import ConvolutionNN
from backend.algorithmic_processing.models.neural_network_models.recurrent_nn_model import RecurrentNN
from backend.algorithmic_processing.models.neural_network_models.temporary_graph_nn_model import GraphNN
from backend.algorithmic_processing.pre_post_processing.input_to_tensor import fen_to_tensor_cnn, fen_to_tensor_rnn, fen_to_tensor_gnn
from backend.algorithmic_processing.pre_post_processing.input_to_tensor import generate_moves_made

from backend.algorithmic_tests.test_algorithm import is_legal_move

# Necessary Fixes: Return The First Legal Move


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

move_to_id, id_to_move = generate_moves_made(input_files); 

num_classes = len(move_to_id); 

cnn_model = ConvolutionNN(num_classes); 
cnn_model.load_state_dict(torch.load("backend/algorithmic_processing/models/trained_models/trained_models_cnn.pth", map_location="cpu")); 
cnn_model.to("cpu"); 
cnn_model.eval(); 

rnn_model = RecurrentNN(num_classes); 
rnn_model.load_state_dict(torch.load("backend/algorithmic_processing/models/trained_models/trained_models_rnn.pth", map_location="cpu")); 
rnn_model.to("cpu"); 
rnn_model.eval(); 

gnn_model = GraphNN(in_features=12, hidden_features=128, class_number=num_classes); 
gnn_model.load_state_dict(torch.load("backend/algorithmic_processing/models/trained_models/trained_models_gnn.pth", map_location="cpu")); 
gnn_model.to("cpu"); 
gnn_model.eval(); 

# -------------------------- Predictions By Each Model

def predict_move_cnn(fen: str) -> str:
    X = fen_to_tensor_cnn(fen).unsqueeze(0).to("cpu"); 
    with torch.no_grad():
        logits = cnn_model(X); 
        move_id = torch.nn.functional.softmax(logits, dim=1); 
    
    move_indices = torch.argsort(move_id, descending=True).flatten().tolist(); 

    for move in move_indices:
        if is_legal_move(fen , id_to_move[move]):
            return id_to_move[move]; 
    
    return "There is no legal move predicted."; 

def predict_move_rnn(fen: str) -> str:
    X = fen_to_tensor_cnn(fen).unsqueeze(0).to("cpu"); 
    with torch.no_grad():
        logits = rnn_model(X); 
        move_id = torch.nn.functional.softmax(logits, dim=1); 
    
    move_indices = torch.argsort(move_id, descending=True).flatten().tolist(); 

    for move in move_indices:
        if is_legal_move(fen, id_to_move[move]):
            return id_to_move[move]; 
    
    return "There is no legal move predicted."; 

# ---------------------------------------- Issues With This Model, [1, 224] vs. [1, 64] Tensor Sizes. 

def predict_move_gnn(fen: str) -> str:

    node_features, adjacency_matrix = fen_to_tensor_gnn(fen); 
    X = node_features.to("cpu"); 


    edge_index = (adjacency_matrix > 0).nonzero(as_tuple=False).t().to("cpu"); 

    with torch.no_grad():
        logits = gnn_model(X,  edge_index); 
        move_id = torch.nn.functional.softmax(logits, dim=1); 
    
    move_indices = torch.argsort(move_id, descending=True).tolist(); 

    for move in move_indices:
        if is_legal_move(fen, id_to_move[move]):
            return id_to_move[move]; 
    
    return "There is no legal move predicted."; 