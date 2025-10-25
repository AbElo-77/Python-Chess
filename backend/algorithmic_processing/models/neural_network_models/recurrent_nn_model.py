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
               './data/training_dataset/page_10.csv',
               './data/training_dataset/page_11.csv', 
               './data/training_dataset/page_12.csv', 
               './data/training_dataset/page_13.csv', 
               './data/training_dataset/page_14.csv', 
               './data/training_dataset/page_15.csv',
               './data/training_dataset/page_16.csv', 
               './data/training_dataset/page_17.csv', 
               './data/training_dataset/page_18.csv', 
               './data/training_dataset/page_19.csv', 
               './data/training_dataset/page_20.csv',
               './data/training_dataset/page_21.csv', 
               './data/training_dataset/page_22.csv', 
               './data/training_dataset/page_23.csv', 
               './data/training_dataset/page_24.csv', 
               './data/training_dataset/page_25.csv',
               './data/training_dataset/page_26.csv', 
               './data/training_dataset/page_27.csv', 
               './data/training_dataset/page_28.csv', 
               './data/training_dataset/page_29.csv', 
               './data/training_dataset/page_30.csv',
               './data/training_dataset/page_31.csv', 
               './data/training_dataset/page_32.csv', 
               './data/training_dataset/page_33.csv', 
               './data/training_dataset/page_34.csv', 
               './data/training_dataset/page_35.csv',
               './data/training_dataset/page_36.csv', 
               './data/training_dataset/page_37.csv', 
               './data/training_dataset/page_38.csv', 
               './data/training_dataset/page_39.csv', 
               './data/training_dataset/page_40.csv',
               './data/training_dataset/page_41.csv', 
               './data/training_dataset/page_42.csv']

move_to_id, id_to_move = generate_moves_made(input_files); 

# ------------------- Areas For Improvement 
# --------------------------------- 

# ------------------- Dual Attention Vision Transformer (DaViT); Modified For 8x8 Board With 12 In Channels
# https://scispace.com/pdf/davit-dual-attention-vision-transformers-1ut6my54.pdf


# MultiLayer Perceptron (MLP); Linear -> GELU() -> Linear
class MLP(torch.nn.Module): 

    def __init__(self, in_features, out_features=None, hidden_features=None, act_func=torch.nn.GELU):
        super().__init__(); 

        out_features = out_features if out_features else in_features; 
        hidden_features = hidden_features if hidden_features else in_features; 

        self.fully_connected_1 = torch.nn.Linear(in_features, hidden_features); 
        self.act_func = act_func; 
        self.fully_connected_2 = torch.nn.Linear(hidden_features, out_features); 

    def forward(self, input): 

        input = self.fully_connected_1(input); 
        input = self.act_func(input); 
        input = self.fully_connected_2(input); 

        return input; 

# 2x2 Kernel For Convolutional Positional Encoding; Input Aggregated by Features Around It
class ConvPosEncoding(torch.nn.Module): 

    def __init__(self, dims, kernel_size=2): 
        super().__init__(); 

        self.conv_proj = torch.nn.Conv2d(dims, dims, kernel_size=kernel_size, 
                                    stride=1, padding=(kernel_size // 2), groups=dims); 

    def forward(self, input, size: tuple): # N == H * W
        B, _, C = input.shape; 
        H, W = size; 

        features = input.transpose(1, 2).view(B, C, H, W); 
        features = self.conv_proj(features); 
        features = features.flatten(2).transpose(1, 2); 

        return input + features; 

# Reduces Board to Patch Embeddings 
class PatchEmbed(torch.nn.Module): 

    def __init__(self, patch_size=[2, 2], in_channels=12, embed_dims=64, overlap=False): 
        super().__init__(); 

        self.patch_size = patch_size; 

        if patch_size[0] == 2: 
            self.conv_patch = torch.nn.Conv2d(in_channels, embed_dims, kernel_size=3, 
                                        stride=patch_size, padding=1); 
            self.norm = torch.nn.LayerNorm(embed_dims); 
        
        if patch_size[0] == 1:
            kernel_size = 2 if overlap else 1; 
            padding = 1 if overlap else 0; 

            self.conv_patch = torch.nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, 
                                        stride=patch_size, padding=padding); 
            self.norm = torch.nn.LayerNorm(in_channels); 

    def forward(self, input, size: tuple): 
        H, W = size; 

        dims = len(input.shape); 
        if dims == 3:
            B, HW, C = input.shape; 
            input = self.norm(input); 
            input = input.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous(); 

        B, C, H, W = input.shape; 
        if W % self.patch_size[1] != 0:
            input = torch.nn.functional.pad(input, (0, self.patch_size[1] - W % self.patch_size[1])); 
        if H % self.patch_size[0] != 0:
            input = torch.nn.functional.pad(input, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0])); 

        input = self.conv_patch(input); 
        new_size = (input.size(2), input.size(3)); 

        input = input.flatten(2).transpose(1, 2); 
        if dims == 4: 
            input = self.norm(input); 

        return input, new_size; 

# ChannelAttention 

# SpatialAttention 

class ChessDaViT(torch.nn.Module): 

    def __init__(self): 
        return; 

# class RecurrentNN(torch.nn.Module): 
#     def __init__(self, class_number): 
#         super().__init__(); 

#         self.convolution = torch.nn.Sequential(
#             torch.nn.Conv2d(12, 64, kernel_size=3, padding=1), 
#             torch.nn.ReLU(), 
#             torch.nn.Conv2d(64, 128, kernel_size=3, padding=1), 
#             torch.nn.ReLU(), 
#             torch.nn.AdaptiveAvgPool2d((8, 8)),
#             torch.nn.Flatten(start_dim=1) 
#         ); 

#         self.lstm = torch.nn.LSTM(input_size=8192, hidden_size=512, num_layers=1, batch_first=True); 

#         self.connections = torch.nn.Sequential(
#             torch.nn.Linear(512, 256), 
#             torch.nn.ReLU(), 
#             torch.nn.Linear(256, class_number) 
#         ); 

#     def forward(self, input_value): 

#         if input_value.dim() == 3:
#             input_value = input_value.unsqueeze(0); 

#         f = self.convolution(input_value);  
#         f = f.unsqueeze(1);  

#         lstm_out, _ = self.lstm(f); 
#         last_hidden = lstm_out[:, -1, :]; 

#         return self.connections(last_hidden); 

if __name__ == "__main__":

# ------------------- Creating A Loss Function with CrossEntropyLoss()

    loss_function = torch.nn.CrossEntropyLoss(); 

    recurrent_model = ChessDaViT(len(move_to_id)).to("cpu"); 
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
