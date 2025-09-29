import torch; 
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

# ------------------- Areas For Improvement 
# --------------------------------- Modify loader to pass multiple FENs for temporal context
# --------------------------------- Deeper Convolutions, Incorporating ResNet() and AdaptiveAvgPool2D()

# ------------------- Simple Recurrent Neural Network Model

class RecurrentNN(torch.nn.Module): 
    def __init__(self, class_number): 
        super().__init__(); 

        self.convolution = torch.nn.Sequential(
            torch.nn.Conv2d(12, 64, kernel_size=3, padding=1), 
            torch.nn.ReLU(), 
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            torch.nn.ReLU(), 
            torch.nn.AdaptiveAvgPool2d((8, 8)),
            torch.nn.Flatten(start_dim=1) 
        ); 

        self.lstm = torch.nn.LSTM(input_size=8192, hidden_size=512, num_layers=1, batch_first=True); 

        self.connections = torch.nn.Sequential(
            torch.nn.Linear(512, 256), 
            torch.nn.ReLU(), 
            torch.nn.Linear(256, class_number) 
        ); 

    def forward(self, input_value): 

        if input_value.dim() == 3:
            input_value = input_value.unsqueeze(0); 

        f = self.convolution(input_value);  
        f = f.unsqueeze(1);  

        lstm_out, _ = self.lstm(f); 
        last_hidden = lstm_out[:, -1, :]; 

        return self.connections(last_hidden); 

if __name__ == "__main__":

# ------------------- Creating A Loss Function with CrossEntropyLoss()

    loss_function = torch.nn.CrossEntropyLoss(); 

    recurrent_model = RecurrentNN(len(move_to_id)).to("cpu"); 
    optimizing_factor = torch.optim.Adam(recurrent_model.parameters(), lr=1e-5); 

# ------------------- Training The Model With DataLoader

    number_of_epochs = 10; 
    batch_training = create_loader(64, input_files, move_to_id); 

    def model_accuracy(batch_training): 
        number_correct = 0; 
        number_total = 0; 
        recurrent_model.eval(); 

        with torch.no_grad(): 

            for X, Y, Z, y in batch_training: 
                Y, y = Y.to("cpu"), y.to("cpu"); 
                model_output = recurrent_model(Y); 
                predictions = model_output.argmax(dim=1); 
                number_correct += (predictions == y).sum().item(); 
                number_total += y.size(0); 
        
        return number_correct / number_total; 

    for epoch in range(number_of_epochs): 
        recurrent_model.train(); 
        total_loss = 0; 

        for X, Y, Z, y in batch_training:    
            Y, y = Y.to("cpu"), y.to("cpu"); 
            optimizing_factor.zero_grad(); 
            logits = recurrent_model(Y); 
            loss = loss_function(logits, y); 
            loss.backward(); 
            optimizing_factor.step(); 
            total_loss += loss.item(); 
        
        accuracy = model_accuracy(batch_training); 
        print(f"Epoch {epoch+1}/{number_of_epochs} - Accuracy: {accuracy:.4f}"); 

    torch.save(recurrent_model.state_dict(), './backend/algorithmic_processing/models/trained_models.pth'); 
