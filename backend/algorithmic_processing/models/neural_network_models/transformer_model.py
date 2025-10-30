import itertools
import torch, timm; 
from timm.layers import DropPath
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
               './data/training_dataset/page_41.csv',]

move_to_id, id_to_move = generate_moves_made(input_files); 

# ------------------- Areas For Improvement 
# --------------------------------- 

# ------------------- Dual Attention Vision Transformer (DaViT); Modified For 8x8 Board With 12 In Channels
# https://scispace.com/pdf/davit-dual-attention-vision-transformers-1ut6my54.pdf

# -----------------------------------------------------------------------------------------------------------

# MultiLayer Perceptron (MLP); Linear -> GELU() -> Linear
class MLP(torch.nn.Module): 

    def __init__(self, in_features, out_features=None, hidden_features=None, act_func=torch.nn.GELU()):
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
        B, N, C = input.shape; 
        H, W = size; 
        assert N == H*W; 

        features = input.transpose(1, 2).reshape(B, C, H, W); 
        features = self.conv_proj(features); 
        features = features.flatten(2).transpose(1, 2); 

        if features.size(1) != input.size(1):
            if features.size(1) > input.size(1):
                features = features[:, :input.size(1), :]
            else:
                pad_tokens = input.size(1) - features.size(1)
                pad = torch.zeros(B, pad_tokens, C, device=input.device, dtype=input.dtype)
                features = torch.cat([features, pad], dim=1)

        return input + features

# Reduces Board to Patch Embeddings 
class PatchEmbed(torch.nn.Module): 

    def __init__(self, patch_size=(2, 2), in_channels=12, embed_dims=64, overlap=False): 
        super().__init__(); 

        self.patch_size = patch_size; 
        self.in_channels = in_channels; 
        self.embed_dims = embed_dims; 

        if self.patch_size[0] == 2: 
            self.conv_patch = torch.nn.Conv2d(
                in_channels, embed_dims,
                kernel_size=3, stride=patch_size, padding=1
            )
            self.norm = torch.nn.LayerNorm(embed_dims); 
        
        if self.patch_size[0] == 1:
            kernel_size = 2 if overlap else 1 
            padding = 1 if overlap else 0 
            self.conv_patch = torch.nn.Conv2d(
                in_channels, embed_dims,
                kernel_size=kernel_size, stride=patch_size, padding=padding
            )
            self.norm = torch.nn.LayerNorm(embed_dims); 

    def forward(self, input, size: tuple): 
        H, W = size; 
        dims = len(input.shape); 

        if dims == 3:
            B, N, C = input.shape; 
            input = input.permute(0, 2, 1).reshape(B, C, H, W); 
        B, C, H, W = input.shape; 
        
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
        pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        if pad_w > 0 or pad_h > 0:
            input = torch.nn.functional.pad(input, (0, pad_w, 0, pad_h))

        input = self.conv_patch(input); 
        new_size = (input.size(2), input.size(3)); 

        input = input.flatten(2).transpose(1, 2); 
        input = self.norm(input); 

        return input, new_size; 

# Channel Attention For Global Attention Across Features
class ChannelAttention(torch.nn.Module): 

    def __init__(self, dim, num_heads): 
        super().__init__(); 

        self.num_heads = num_heads; 
        self.scale  = (dim // num_heads) ** -0.5; 

        self.proj = torch.nn.Linear(dim, dim); 
        self.QKV = torch.nn.Linear(dim, dim * 3); 

    def forward(self, input): 
        B, N, C = input.shape; 

        if C % self.num_heads != 0:
            raise RuntimeError(f"Embedding dim {C} not divisible by num_heads {self.num_heads}");  
    
        head_dim = C // self.num_heads; 

        QKV = self.QKV(input).view(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4); 

        query, key, value = QKV[0], QKV[1], QKV[2]; 
        key *= self.scale; 
        attention = key.transpose(-1, -2) @ value; 
        attention = attention.softmax(dim=-1); 

        input = (attention @ query.transpose(-1, -2)).transpose(-1, -2); 
        input = self.proj(input.transpose(1, 2).reshape(B, N, C)); 

        return input; 

# Window-Based Spatial Attention For Local Attention Among Tokens
class SpatialAttention(torch.nn.Module):

    def __init__(self, dim, num_heads): 
        super().__init__(); 

        self.num_heads = num_heads; 
        self.scale  = (dim // num_heads) ** -0.5; 

        self.proj = torch.nn.Linear(dim, dim); 
        self.QKV = torch.nn.Linear(dim, dim * 3); 

    def forward(self, input): 
        B, N, C = input.shape; 
        
        if C % self.num_heads != 0:
            raise RuntimeError(f"Embedding dim {C} not divisible by num_heads {self.num_heads}");  
        
        head_dim = C // self.num_heads
        QKV = self.QKV(input).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4); 

        query, key, value = QKV[0], QKV[1], QKV[2]; 
        key *= self.scale; 
        attention = query @ key.transpose(-2, -1); 
        attention = attention.softmax(dim=-1); 

        input = (attention @ value).transpose(1, 2).reshape(B, N, C);
        input = self.proj(input); 

        return input; 

# Channel Attention Block For A Sequence of Channel Attending
class ChanAttenBlock(torch.nn.Module): 

    def __init__(self, dims, num_heads, MLP_ratio, drop_path=0.1, feed_forward=True): 
        super().__init__()
        
        self.dims = dims; 
        self.conv_pos = torch.nn.ModuleList([
            ConvPosEncoding(dims, kernel_size=2),
            ConvPosEncoding(dims, kernel_size=2)
        ])
        
        self.attention = ChannelAttention(dims, num_heads); 
        self.drop_path = DropPath(drop_path) if drop_path > 0 else None; 
        
        self.ff = feed_forward; 
        self.MLP = MLP(dims, hidden_features=int(dims * MLP_ratio)); 
        
        self.norm1 = torch.nn.LayerNorm(dims); 
        self.norm2 = torch.nn.LayerNorm(dims); 

    def forward(self, input, size: tuple):
        residual = input; 
        
        input = self.conv_pos[0](input, size); 
        input = self.norm1(input); 
        input = self.attention(input); 
        
        if self.drop_path is not None:
            input = residual + self.drop_path(input); 
        else:
            input = residual + input; 
    
        if self.ff:
            residual = input; 
            input = self.conv_pos[1](input, size); 
            input = self.norm2(input); 
            input = self.MLP(input); 
            if self.drop_path is not None:
                input = residual + self.drop_path(input); 
            else:
                input = residual + input; 

        return input, size; 


# Spatial Attention Block For A Sequence of Spatial Attending
class SpatAttenBlock(torch.nn.Module):

    def __init__(self, dims, num_heads, MLP_ratio, drop_path=0., feed_forward=True, window_size=2): 
        super().__init__(); 
        
        self.dims = dims; 
        self.window_size = window_size; 
        self.conv_pos = torch.nn.ModuleList([
            ConvPosEncoding(dims, kernel_size=2),
            ConvPosEncoding(dims, kernel_size=2)
        ])
        
        self.attention = SpatialAttention(dims, num_heads); 
        self.drop_path = DropPath(drop_path) if drop_path > 0 else None; 
        
        self.ff = feed_forward; 
        self.MLP = MLP(dims, hidden_features=int(dims * MLP_ratio)); 
        
        self.norm1 = torch.nn.LayerNorm(dims); 
        self.norm2 = torch.nn.LayerNorm(dims); 
    
    def window(self, input, window_size): 
        B, H, W, C = input.shape; 
    
        assert H % window_size == 0, f"Input height {H} not divisible by window size {window_size}"
        assert W % window_size == 0, f"Input width {W} not divisible by window size {window_size}"
        
        input = input.view(B, H // window_size, window_size, W // window_size, window_size, C); 
        input_windows = input.permute(0, 1, 3, 2, 4, 5).contiguous()
        input_windows = input_windows.view(-1, window_size, window_size, C); 
        
        return input_windows; 

    def window_r(self, input, window_size, H_new, W_new): 
        B = int(input.shape[0] / ((H_new * W_new) / (window_size * window_size))); 
        C = input.shape[-1]; 
        
        input_r = input.view(B, H_new // window_size, W_new // window_size, window_size, window_size, C); 
        input_r = input_r.permute(0, 1, 3, 2, 4, 5).contiguous(); 
        input_r = input_r.view(B, H_new, W_new, C); 
        
        return input_r; 

    def forward(self, input, size): 
        B, N, C = input.shape; 
        H, W = size; 

        residual = input; 
        
        input = self.norm1(input); 
        input = self.conv_pos[0](input, size); 
        input = input.view(B, H, W, C); 
        
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        if pad_w > 0 or pad_h > 0:
            input = torch.nn.functional.pad(input, (0, 0, 0, pad_w, 0, pad_h))
        
        H_new, W_new = input.shape[1:3]; 
        
        input_windows = self.window(input, self.window_size); 
        input_windows = input_windows.view(-1, self.window_size * self.window_size, C); 
        

        attention_windows = self.attention(input_windows); 
        attention_windows = attention_windows.view(-1, self.window_size, self.window_size, C); 
        
        input = self.window_r(attention_windows, self.window_size, H_new, W_new); 
        
        if H != H_new or W != W_new:
            input = input[:, :H, :W, :].contiguous(); 
        
        input = input.view(B, H*W, C);  
        
        if self.drop_path is not None:
            input = residual + self.drop_path(input); 
        else:
            input = residual + input; 
            
        if self.ff:
            residual = input; 
            input = self.norm2(input); 
            input = self.MLP(input); 
            if self.drop_path is not None:
                input = residual + self.drop_path(input); 
            else:
                input = residual + input; 

        input = self.conv_pos[1](input, size); 
        if self.ff: 
            proj = self.norm2(input); 
            proj = self.MLP(input); 
            
            if self.drop_path: 
                input = input + self.drop_path(proj); 

        return input, size;  

# Sequential Allowing Multiple Inputs / Outputs
class MultipleSequential(torch.nn.Sequential):

    def forward(self, *inputs):

        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs); 
            else:
                inputs = module(inputs); 
        
        return inputs; 


class ChessDaViT(torch.nn.Module): 

    def __init__(self, in_channels=12, num_classes=len(move_to_id), 
                 depths=[1, 1, 2, 1], drop_rate=0.1, patch_size=[1, 1], 
                 embed_dims=[128, 256, 512, 1024], num_heads=[4, 8, 16, 32], image_size=8): 
        super().__init__()

        attention_types = ('spatial', 'channel'); 
        architecture = [[index] * item for index, item in enumerate(depths)]; 
        
        assert image_size == 8, "Input image size must be 8x8 for chess board"
        assert len(embed_dims) == len(num_heads), "Must have same number of embedding dims and attention heads"
        
        self.architecture = architecture; 
        self.num_classes = num_classes; 
        self.image_size = image_size; 
        self.embed_dims = embed_dims; 
        self.num_heads = num_heads; 

        drop_rates = [x.item() for x in torch.linspace(0, drop_rate, 2 * len(list(itertools.chain(*self.architecture))))]
        self.patch_embedings = torch.nn.ModuleList([
            PatchEmbed(
                patch_size=patch_size,
                in_channels=in_channels if i == 0 else embed_dims[i - 1],
                embed_dims=embed_dims[i],
                overlap=False
            )
            for i in range(len(embed_dims))
        ])

        main_blocks = []
        for block_id, block_param in enumerate(self.architecture):
            layer_offset_id = len(list(itertools.chain(*self.architecture[:block_id]))); 

            block = torch.nn.ModuleList([
                MultipleSequential(*[
                    ChanAttenBlock(
                        self.embed_dims[item],
                        self.num_heads[item], 
                        4.0,
                        drop_path=drop_rates[2 * (layer_id + layer_offset_id) + attention_id],
                    ) if attention_type == 'channel' else
                    SpatAttenBlock(
                        self.embed_dims[item],
                        self.num_heads[item],
                        4.0,
                        drop_path=drop_rates[2 * (layer_id + layer_offset_id) + attention_id],
                    ) if attention_type == 'spatial' else None
                    for attention_id, attention_type in enumerate(attention_types)]
                ) for layer_id, item in enumerate(block_param)
            ])
            main_blocks.append(block); 
        self.main_blocks = torch.nn.ModuleList(main_blocks); 

        self.norm_layer = torch.nn.LayerNorm(normalized_shape=self.embed_dims[-1]); 
        self.head = torch.nn.Linear(in_features=self.embed_dims[-1], out_features=num_classes); 
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1); 

    def forward(self, input):

        if input.dim() != 4:
            raise ValueError("")
            
        B, C, H, W = input.shape
        
        input, size = self.patch_embedings[0](input, (input.size(2), input.size(3))); 
        
        features = [input]; 
        sizes = [size]; 
        branches = [0]; 

        for block_index, block_param in enumerate(self.architecture):
            branch_ids = sorted(set(block_param)); 

            for branch_id in branch_ids:
                if branch_id not in branches:
                    input, size = self.patch_embedings[branch_id](features[-1].flatten(2), sizes[-1]); 
                    features.append(input); 
                    sizes.append(size); 
                    branches.append(branch_id); 
            
            for layer_index, branch_id in enumerate(block_param):
                features[branch_id], _ = self.main_blocks[block_index][layer_index](features[branch_id], sizes[branch_id]); 

        features[-1] = self.avg_pool(features[-1].transpose(1, 2)); 
        features[-1] = torch.flatten(features[-1], 1); 

        input = self.norm_layer(features[-1]); 
        input = self.head(input); 

        return input; 

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); 

    transformer_model = ChessDaViT().to(device); 
    optimizing_factor = torch.optim.Adam(transformer_model.parameters(), lr=1e-5); 

# ------------------- Training The Model With DataLoader

    number_of_epochs = 10; 
    batch_training = create_loader(64, input_files, move_to_id); 

    def model_accuracy(batch_training): 
        number_correct = 0; 
        number_total = 0; 
        transformer_model.eval(); 

        with torch.no_grad(): 

            for X, Y, Z, y in batch_training: 

                Y, y = Y.to(device), y.to(device).view(-1); 
                model_output = transformer_model(Y); 

                predictions = model_output.argmax(dim=1); 
                number_correct = number_correct + (predictions == y).sum().item(); 

                number_total = number_total + y.size(0); 
        
        return number_correct / number_total; 

    for epoch in range(number_of_epochs): 
        transformer_model.train(); 
        total_loss = 0; 

        for X, Y, Z, y in batch_training:    

            Y, y = Y.to(device), y.to(device).view(-1); 
            
            optimizing_factor.zero_grad(); 
            logits = transformer_model(Y); 
            
            loss = loss_function(logits, y); 
            loss.backward(); 
            
            optimizing_factor.step(); 
            total_loss = total_loss + loss.item(); 
        
        accuracy = model_accuracy(batch_training); 
        print(f"Epoch {epoch+1}/{number_of_epochs} - Accuracy: {accuracy:.4f}"); 

    torch.save(transformer_model.state_dict(), './backend/algorithmic_processing/models/trained_models.pth'); 
