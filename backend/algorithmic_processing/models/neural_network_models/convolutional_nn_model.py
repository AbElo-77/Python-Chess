import os 
import torch
# import boto3
from torch.amp import autocast, GradScaler
from timm.layers import DropPath, trunc_normal_
from backend.algorithmic_processing.pre_post_processing.input_to_tensor import generate_moves_made, create_loader

# s3 = boto3.client("s3")
# S3_BUCKET = ""

# ------------------- Areas For Improvement 
# --------------------------------- Take Input From Past Three FENs To Establish Temporal Context
# --------------------------------- Modify ConvNeXt and Blocks For Better Global Attention 

# ------------------- ConvNeXt Model Modified For 8x8 Board With 12 In Channels
# https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf

# -----------------------------------------------------------------------------------------------------------

# Block Featuring Depthwise and Pointwise Convolutions
class Block(torch.nn.Module): 
    
    def __init__(self, dim, drop_path=0, layer_scale=1e-6):
        super().__init__()

        self.conv_dw = torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim) 
        self.norm = LayerNorm(dim, eps=1e-6) 

        self.conv_pw = torch.nn.Linear(dim, dim * 4) 
        self.act_func = torch.nn.GELU() 
        self.conv_pw_r = torch.nn.Linear(dim * 4, dim) 

        self.gamma = torch.nn.Parameter(layer_scale * torch.ones((dim)), requires_grad=True) 
        self.drop_path = DropPath(drop_path) if drop_path > 0 else None 

    def forward(self, input):
        original_input = input 

        input = self.conv_dw(input) 
        input = input.permute(0, 2, 3, 1) 

        input = self.norm(input) 

        input = self.conv_pw(input) 
        input = self.act_func(input) 
        input = self.conv_pw_r(input) 

        input = self.gamma * input if self.gamma is not None else input 
        input = input.permute(0, 3, 1, 2) 

        if self.drop_path is not None:
            input = original_input + self.drop_path(input) 
        else:
            input = original_input + input 
        return input 


# Custom LayerNorm For Alternating Input Dims 
class LayerNorm(torch.nn.Module): 

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"): 
        super().__init__() 

        self.weights = torch.nn.Parameter(torch.ones(normalized_shape)) 
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape)) 
        self.eps = eps 

        self.data_format = data_format 
        self.normalized_shape = (normalized_shape, ) 

    def forward(self, input): 
        if self.data_format == "channels_last":
            return torch.nn.functional.layer_norm(input, self.normalized_shape, 
                                                  self.weights, self.bias, self.eps) 
        else: 
            u = input.mean(1, keepdim=True) 
            s = (input - u).pow(2).mean(1, keepdim=True) 
            input = (input - u) / torch.sqrt(s + self.eps) 
            input = self.weights[:, None, None] * input + self.bias[:, None, None] 
            return input 

class ChessConvNeXt(torch.nn.Module): 

    def __init__(self,  num_classes, in_channels=12, 
                 depths=[3, 3, 9, 3], dims=[64, 128, 256, 512], drop_rate=0.1, layer_scale=1e-6, head_scale=1.): 
        super().__init__() 

        
        self.downsample_layers = torch.nn.ModuleList() 
        stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, dims[0], kernel_size=2, stride=2), 
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        ) 
        self.downsample_layers.append(stem) 

        for i in range(3): 
            if i == 2:
                conv = torch.nn.Conv2d(dims[i], dims[i+1], kernel_size=1, stride=1) 
            else:
                conv = torch.nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2) 
            downsample_layer = torch.nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                conv,
            )
            self.downsample_layers.append(downsample_layer) 

        self.stages = torch.nn.ModuleList() 
        drop_rates=[x.item() for x in torch.linspace(0, drop_rate, sum(depths))] 
        cur = 0 
        for i in range(4):
            stage = torch.nn.Sequential(
                *[Block(dim=dims[i], drop_path=drop_rates[cur + j], 
                layer_scale=layer_scale) for j in range(depths[i])]
            )
            self.stages.append(stage) 
            cur += depths[i] 

        self.norm = torch.nn.LayerNorm(dims[-1], eps=1e-6) 
        self.head = torch.nn.Linear(dims[-1], num_classes) 

        self.apply(self._init_weights) 
        self.head.weight.data.mul_(head_scale) 
        self.head.bias.data.mul_(head_scale) 

    def _init_weights(self, m):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            trunc_normal_(m.weight, std=.02) 
            torch.nn.init.constant_(m.bias, 0) 

    def forward_features(self, input):
        for i in range(4):
            input = self.downsample_layers[i](input) 
            input = self.stages[i](input) 
        
        return self.norm(input.mean([-2, -1])) 

    def forward(self, input): 
        input = self.forward_features(input) 
        input = self.head(input) 

        return input 

# class ConvolutionNN(torch.nn.Module):
#     def __init__(self, class_number):
#         super().__init__() 

#         self.convolution = torch.nn.Sequential(
#             torch.nn.Conv2d(12, 64, kernel_size=3, padding=1), 
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(128, 256, kernel_size=3, padding=1), 
#             torch.nn.ReLU(),
#             torch.nn.AdaptiveAvgPool2d((8, 8)), 
#             torch.nn.Flatten()
#         ) 

#         self.connections = torch.nn.Sequential(
#             torch.nn.Linear(256 * 8 * 8, 512),  
#             torch.nn.ReLU(), 
#             torch.nn.Linear(512, class_number)
#         ) 

#     def forward(self, input_value):
#         return self.connections(self.convolution(input_value)) 

if __name__ == '__main__':

    input_files = ['./data/training_dataset/all_files.csv']
    move_to_id, id_to_move = generate_moves_made(input_files) 

    loss_function = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    convolution_model = ChessConvNeXt(num_classes=len(move_to_id)).to(device)
    optimizing_factor = torch.optim.Adam(convolution_model.parameters(), lr=1e-5)

    scaler = GradScaler()

    start_epoch = 0
    if os.path.exists("checkpoint.pt"):
        ckpt = torch.load("checkpoint.pt", map_location=device)
        convolution_model.load_state_dict(ckpt["model"])
        optimizing_factor.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    number_of_epochs = 10
    batch_training = create_loader(64, input_files, move_to_id)

    def model_accuracy(loader):
        convolution_model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for X, Y, Z, y in loader:
                X, y = X.to(device), y.to(device)
                with autocast('cuda'):
                    model_output = convolution_model(X)
                predictions = model_output.argmax(dim=1)
                correct += (predictions == y).sum().item()
                total += y.size(0)

        return correct / total

    for epoch in range(start_epoch, number_of_epochs):
        convolution_model.train()
        total_loss = 0

        for X, Y, Z, y in batch_training:
            X, y = X.to(device), y.to(device)
            optimizing_factor.zero_grad()
            with autocast('cuda'):
                logits = convolution_model(X)
                loss = loss_function(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizing_factor)
            scaler.update()

            total_loss += loss.item()

        accuracy = model_accuracy(batch_training)
        print(f"Epoch {epoch+1}/{number_of_epochs} | Loss: {total_loss:.4f} | Acc: {accuracy:.4f}")

        torch.save({
            "epoch": epoch,
            "model": convolution_model.state_dict(),
            "optimizer": optimizing_factor.state_dict(),
            "scaler": scaler.state_dict()
        }, f"../trained_models/checkpoints/checkpoint_{epoch}.pt")


        # s3.upload_file(
        #     "checkpoint.pt",
        #     S3_BUCKET,
        #     f"checkpoints/chess_convnext_epoch_{epoch}.pt"
        # )

