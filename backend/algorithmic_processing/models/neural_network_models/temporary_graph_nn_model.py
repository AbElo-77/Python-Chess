import torch
from backend.algorithmic_processing.pre_post_processing.input_to_tensor import generate_moves_made, create_loader

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

# ------------------- Simple Graph Neural Network Model 

class GraphNN(torch.nn.Module):
    def __init__(self, in_features, hidden_features, class_number):
        super().__init__(); 

        self.W1 = torch.nn.Linear(in_features, hidden_features); 
        self.W2 = torch.nn.Linear(hidden_features, hidden_features); 
        
        self.fc_one = torch.nn.Linear(hidden_features, hidden_features // 2); 
        self.fc_two = torch.nn.Linear(hidden_features // 2, class_number); 

    def forward(self, X, A):

        if X.dim() == 2:
            X = X.unsqueeze(0); 
            A = A.unsqueeze(0); 

        batch_size, num_nodes, _ = X.size(); 

        H = self.W1(X); 
        H = torch.matmul(A, H); 
        H = torch.nn.functional.relu(H); 

        H = self.W2(H);             
        H = torch.matmul(A, H); 
        H = torch.nn.functional.relu(H); 

        H = H.mean(dim=1);          

        out = self.fc_one(H);        
        return out; 

if __name__ == "__main__":

# ------------------- Creating A Loss Function with CrossEntropyLoss()

    loss_function = torch.nn.CrossEntropyLoss(); 

    graph_model = GraphNN(in_features=12, hidden_features=128, class_number=len(move_to_id)).to("cpu"); 
    optimizing_factor = torch.optim.Adam(graph_model.parameters(), lr=1e-5); 

# ------------------- Training The Model With DataLoader

    number_of_epochs = 10; 
    batch_training = create_loader(64, input_files, move_to_id); 

    def model_accuracy_gnn(batch_training): 
        number_correct, number_total = 0, 0; 
        graph_model.eval(); 
        
        with torch.no_grad(): 
            for X, Y, Z, y in batch_training: 
                y = y.to("cpu"); 
                batch_predictions = []; 

                for node_features, adjacency_matrix in Z:  
                    node_features = node_features.to("cpu"); 
                    adjacency_matrix = adjacency_matrix.to("cpu"); 

                    logits = graph_model(node_features, adjacency_matrix); 
                    pred = logits.argmax(dim=1); 
                    batch_predictions.append(pred); 

                batch_predictions = torch.cat(batch_predictions); 
                number_correct += (batch_predictions == y).sum().item(); 
                number_total += y.size(0); 

        return number_correct / number_total if number_total > 0 else 0; 

    for epoch in range(number_of_epochs): 
        graph_model.train(); 
        total_loss = 0; 
        
        for X, Y, Z, y in batch_training:    
            y = y.to("cpu"); 
            optimizing_factor.zero_grad(); 
            
            batch_logits = []; 
            batch_loss = 0; 

            for i, (node_features, adjacency_matrix) in enumerate(Z): 
                node_features = node_features.to("cpu"); 
                adjacency_matrix = adjacency_matrix.to("cpu"); 

                logits = graph_model(node_features, adjacency_matrix)  
                loss = loss_function(logits, y[i].unsqueeze(0)) / len(Z);  
                loss.backward(); 

                batch_logits.append(logits); 
                batch_loss += loss.item(); 

            optimizing_factor.step(); 
            total_loss += batch_loss; 

        accuracy = model_accuracy_gnn(batch_training); 
        print(f"Epoch {epoch+1}/{number_of_epochs} - Loss: {total_loss:.4f} - Accuracy: {accuracy:.4f}"); 

    torch.save(graph_model.state_dict(), './backend/algorithmic_processing/models/trained_models.pth'); 
