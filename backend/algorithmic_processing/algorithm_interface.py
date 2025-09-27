import torch

from backend.algorithmic_processing.models.neural_network_models.convolutional_nn_model import ConvolutionNN
from backend.algorithmic_processing.pre_post_processing.input_to_tensor import fen_to_tensor_cnn
from backend.algorithmic_processing.pre_post_processing.input_to_tensor import generate_moves_made


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

import json
with open("move_to_id.json", "r") as f:
    move_to_id = json.load(f); 
id_to_move = {v: k for k, v in move_to_id.items()}

number_of_moves = len(move_to_id); 

model = ConvolutionNN(num_classes=number_of_moves); 
model.load_state_dict(torch.load("chess_model.pt", map_location="cpu")); 
model.to("cpu"); 
model.eval(); 


def predict_move_cnn(fen: str) -> str:
    X = fen_to_tensor_cnn(fen).unsqueeze(0).to("cpu"); 

    with torch.no_grad():
        logits = model(X); 
        probs = torch.nn.functional.softmax(logits, dim=1); 

        move_id = torch.argmax(probs, dim=1).item(); 
        move = id_to_move[move_id]; 

    return move; 
