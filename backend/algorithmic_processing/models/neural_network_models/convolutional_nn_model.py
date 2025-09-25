import torch
from backend.algorithmic_processing.pre_post_processing.input_to_tensor import create_loader

# ------------------- Areas For Improvement 
# --------------------------------- Take Input From Past Three FENs To Establish Temporal Context
# --------------------------------- Deeper Convolutions, Incorporating ResNet() and AdaptiveAvgPool2D()

# ------------------- Simple Convolution Neural Network Model, Producing <output_number> Move Predictions

class ConvolutionNN(torch.nn.Module):
    def __init__(self, output_number):
        super().__init__(); 

        self.convolution = torch.nn.Sequential(
            torch.nn.Conv2d(12, 64, kernel_size=1, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=1, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=1, padding=3), 
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )

        self.connections = torch.nn.Sequential(
            torch.nn.Linear(8192, 512), 
            torch.nn.ReLU(), 
            torch.nn.Linear(512, output_number)
        )

    def make_forward_pass(self, input_value):
        return self.connections(self.convolution(input_value)); 

# ------------------- Creating A Loss Function with CrossEntropyLoss()

loss_function = torch.nn.CrossEntropyLoss(); 

convolution_model = ConvolutionNN(10).to("cpu"); 
optimizing_factor = torch.optim.Adam(convolution_model.parameters(), lr=1e-5)

# ------------------- Training The Model With DataLoader

number_of_epochs = 10; 
batch_training = create_loader(64); 

def model_accuracy(batch_training): 
    number_correct, number_total = 0; 
    convolution_model.eval(); 

    for X, y in batch_training: 
        X, y = X.to("cpu"), y.to("cpu"); 

        model_output = convolution_model(X); 

        number_correct += (model_output == y).sum().item(); 
        number_total += y.size(0); 
    
    return number_correct / number_total; 

for epoch in range(number_of_epochs): 
    convolution_model.train(); 
    total_loss = 0; 

    for X, y in batch_training:    
        X, y = X.to("cpu"), y.to("cpu"); 

        optimizing_factor.zero_grad(); 

        logits = convolution_model(X); 

        loss = loss_function(logits, y); 
        loss.backward(); 
        loss_function.step(); 

        total_loss += loss.item(); 

    accuracy = model_accuracy(batch_training); 
    print(f"Epoch {epoch+1}/{number_of_epochs} - Accuracy: {accuracy:.4f}"); 

trained_convolution_model = torch.save(convolution_model.state_dict(), 
                                       './backend/algorithmic_processing/models/trained_models'); 
