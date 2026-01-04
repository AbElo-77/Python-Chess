import os
import torch, torch_geometric.nn
import boto3
from torch.cuda.amp import autocast, GradScaler
from timm.layers import DropPath
from backend.algorithmic_processing.pre_post_processing.input_to_tensor import generate_moves_made, create_loader

s3 = boto3.client("s3")
S3_BUCKET = ""

input_files = "/home/ubuntu/data/all_files.csv"
move_to_id, id_to_move = generate_moves_made(input_files) 

# ------------------- Spatio-Temporal Graph Neural Network (STGNN) Modified For 8x8 Board With 12 In Channels 
# https://epubs.siam.org/doi/10.1137/1.9781611976700.82
# https://www.sciencedirect.com/science/article/abs/pii/S1566253525000703

# -----------------------------------------------------------------------------------------------------------

class GraphNorm(torch.nn.Module): 

    def __init__(self, embed_dims, k=8): 
        super().__init__()
        
        self.embed_dims = embed_dims
        self.k = k
        self.norm = None

    def forward(self, input):
        X, A = input
        
        if A.dim() == 2:
            A = A.unsqueeze(0)
        
        while A.dim() > 3:
            A = A.squeeze(0)
        
        if A.dim() < 3:
            A = A.unsqueeze(0)
            
        B = A.shape[0]
        N = A.shape[1]
        
        degrees = A.sum(dim=2, keepdim=True) 
        degrees_sqrt_inv = torch.pow(degrees + 1e-8, -0.5)  
        degrees_sqrt_inv[torch.isinf(degrees_sqrt_inv)] = 0.0
        
        A_norm = degrees_sqrt_inv * A * degrees_sqrt_inv.transpose(1, 2)
        
        if X.dim() == 4:

            C = X.shape[-1]
            if self.norm is None or tuple(self.norm.normalized_shape) != (C,):
                self.norm = torch.nn.LayerNorm(C)
            X_norm = self.norm(X)
        else:
            X_norm = X

        if A_norm.shape[0] == 1:
            A_norm = A_norm.squeeze(0)
        
        return (X_norm, A_norm)

class TemporalConvolution(torch.nn.Module): 

    def __init__(self, in_channels, out_channels, embed_dims, kernel_size=3): 
        super().__init__()

        pad = (0, kernel_size // 2)
        self.conv_1 = torch.nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=pad)
        self.conv_2 = torch.nn.Conv2d(out_channels, out_channels, (1, kernel_size), padding=pad)

        self.norm = torch.nn.LayerNorm(embed_dims)

    def forward(self, input, size: tuple): 
        X, A = input
        
        # Shape: B, N, T, C -> B, C, N, T
        X = X.permute(0, 3, 1, 2)

        X = self.conv_1(X)
        X = X.permute(0, 2, 3, 1)
        X = self.norm(X).permute(0, 3, 1, 2)
        
        B, C, N, T = X.shape

        if A.dim() == 2:
            A_batch = A.unsqueeze(0)
        else:
            A_batch = A

        deg = A_batch.sum(dim=-1, keepdim=True) 
        deg_inv = torch.pow(deg + 1e-8, -1.0)
        deg_inv[torch.isinf(deg_inv)] = 0.0


        X_nodes = X.permute(0, 2, 3, 1).mean(dim=2)  

        messages = []
        for b in range(B):
            A_b = A_batch[b]
            D_inv_b = torch.diag_embed(deg_inv[b].squeeze(-1)) if deg_inv.size(0) > 0 else torch.eye(N, device=A_b.device)
            A_norm_b = D_inv_b @ A_b @ D_inv_b
            m = A_norm_b @ X_nodes[b]
            messages.append(m)

        M = torch.stack(messages, dim=0) 
        X_msg = M.unsqueeze(2).expand(-1, -1, T, -1).permute(0, 3, 1, 2)  


        X = self.conv_2(X_msg)

        X = X.permute(0, 2, 3, 1)

        return X

class SpatialAttention(torch.nn.Module): 

    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, embed_dims, mlp=True):
        super().__init__()

        self.num_heads = num_heads 
        self.mlp = mlp


        self.relu = torch.nn.ReLU()

        self.lin_q = torch.nn.Linear(in_channels, out_channels)
        self.lin_k = torch.nn.Linear(in_channels, out_channels)
        self.lin_v = torch.nn.Linear(in_channels, out_channels)
        if mlp:
            self.MLP = torch_geometric.nn.MLP([in_channels, hidden_channels, out_channels])
        self.drop_path = DropPath(0.1)

    def forward(self, input):
        X, A = input

        if A.dim() == 2:
            A = A.unsqueeze(0)

        B, N, T, C = X.shape

        X_t = X.mean(dim=2)  
        outputs = []
        for b in range(B):
            x_b = X_t[b]  
            a_b = A[b] if A.dim() == 3 else A

            Q = self.lin_q(x_b)  
            K = self.lin_k(x_b) 
            V = self.lin_v(x_b)

            scores = Q @ K.t() / (K.shape[-1] ** 0.5) 
            mask = (a_b > 0).to(scores.dtype)

            mask = mask + torch.eye(N, device=mask.device, dtype=mask.dtype)
            scores = scores * mask - (1.0 - mask) * 1e9

            attn = torch.softmax(scores, dim=-1)
            h = attn @ V  

            if self.mlp:
                x_proj = self.MLP(x_b)
                h = h + self.drop_path(x_proj)

            outputs.append(h)

        H = torch.stack(outputs, dim=0)  
        H = H.unsqueeze(2).expand(-1, -1, T, -1) 

        return H, A

class ChessSTGNN(torch.nn.Module): 

    def __init__(self, in_channels=64, hidden_channels=128, out_channels=256, 
                 num_heads=3, embed_dims=4, kernel_size=3, mlp=True, num_classes=len(move_to_id)): 
        super().__init__()
        
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        
        self.graph_norm = GraphNorm(embed_dims)
        
        self.temporal_conv_1 = TemporalConvolution(
            in_channels=in_channels, 
            out_channels=hidden_channels, 
            embed_dims=embed_dims, 
            kernel_size=kernel_size
        )
        
        self.spatial_attn_1 = SpatialAttention(
            in_channels=hidden_channels, 
            hidden_channels=hidden_channels, 
            out_channels=hidden_channels, 
            num_heads=num_heads, 
            embed_dims=embed_dims, 
            mlp=mlp
        )
        
        self.temporal_conv_2 = TemporalConvolution(
            in_channels=hidden_channels, 
            out_channels=hidden_channels, 
            embed_dims=embed_dims, 
            kernel_size=kernel_size
        )
        
        self.spatial_attn_2 = SpatialAttention(
            in_channels=hidden_channels, 
            hidden_channels=hidden_channels, 
            out_channels=out_channels, 
            num_heads=num_heads, 
            embed_dims=embed_dims, 
            mlp=mlp
        )

        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc_head = torch.nn.Sequential(
            torch.nn.Linear(out_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(self, X, A):

        if X.dim() == 2:
            X = X.unsqueeze(0).unsqueeze(2)  
        elif X.dim() == 3:
            
            X = X.unsqueeze(2)
        
        if A.dim() == 2:
            A = A.unsqueeze(0) 
        
        X, A = self.graph_norm((X, A))
        
        X = self.temporal_conv_1((X, A), size=X.shape)
        X, A = self.spatial_attn_1((X, A))
        
        X = self.temporal_conv_2((X, A), size=X.shape)
        X, A = self.spatial_attn_2((X, A))
        
        B, N, T, C = X.shape
        X = X.permute(0, 3, 1, 2)  
        X = X.reshape(B, C, N * T)  
        X = X.mean(dim=2)  
        
        logits = self.fc_head(X)  
        
        return logits
    
    def encode(self, X, A):

        if X.dim() == 2:
            X = X.unsqueeze(0).unsqueeze(2)
        elif X.dim() == 3:
            X = X.unsqueeze(2)
        
        if A.dim() == 2:
            A = A.unsqueeze(0)
        
        X, A = self.graph_norm((X, A))
        X = self.temporal_conv_1((X, A), size=X.shape)
        X, A = self.spatial_attn_1((X, A))
        X = self.temporal_conv_2((X, A), size=X.shape)
        X, A = self.spatial_attn_2((X, A))
        
        B, N, T, C = X.shape
        X = X.permute(0, 3, 1, 2)
        X = X.reshape(B, C, N * T)
        X = X.mean(dim=2)
        
        return X
        

# class GraphNN(torch.nn.Module):
#     def __init__(self, in_features, hidden_features, class_number):
#         super().__init__() 

#         self.W1 = torch.nn.Linear(in_features, hidden_features) 
#         self.W2 = torch.nn.Linear(hidden_features, hidden_features) 
        
#         self.fc_one = torch.nn.Linear(hidden_features, hidden_features // 2) 
#         self.fc_two = torch.nn.Linear(hidden_features // 2, class_number) 

#     def forward(self, X, A):

#         if X.dim() == 2:
#             X = X.unsqueeze(0) 
#             A = A.unsqueeze(0) 

#         batch_size, num_nodes, _ = X.size() 

#         H = self.W1(X) 
#         H = torch.matmul(A, H) 
#         H = torch.nn.functional.relu(H) 

#         H = self.W2(H)             
#         H = torch.matmul(A, H) 
#         H = torch.nn.functional.relu(H) 

#         H = H.mean(dim=1) 

#         out = self.fc_one(H) 
#         out = torch.nn.functional.relu(out)
#         out = self.fc_two(out) 
#         return out 

if __name__ == "__main__":


    loss_function = torch.nn.CrossEntropyLoss() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph_model = ChessSTGNN(
        in_channels=12, 
        hidden_channels=128, 
        out_channels=128,
        num_heads=8,
        embed_dims=128,
        kernel_size=3,
        mlp=True,
        num_classes=len(move_to_id)
    ).to(device) 
    optimizing_factor = torch.optim.Adam(graph_model.parameters(), lr=1e-5) 

    scaler = GradScaler('cuda')

    start_epoch = 0
    if os.path.exists("checkpoint.pt"):
        ckpt = torch.load("checkpoint.pt", map_location=device)
        graph_model.load_state_dict(ckpt["model"])
        optimizing_factor.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    number_of_epochs = 10 
    batch_training = create_loader(64, input_files, move_to_id) 

    def model_accuracy_gnn(batch_training): 
        number_correct, number_total = 0, 0 
        graph_model.eval() 
        
        with torch.no_grad(): 
            for _, _, Z, y in batch_training: 
                y = y.to(device)

                node_features_batch, adjacency_batch = Z
                node_features_batch = node_features_batch.to(device)
                adjacency_batch = adjacency_batch.to(device)
                
                with autocast('cuda'):
                    logits = graph_model(node_features_batch, adjacency_batch)
                    prediction = logits.argmax(dim=1)
                
                number_correct += (prediction == y).sum().item() 
                number_total += y.size(0) 

        return number_correct / number_total if number_total > 0 else 0 

    for epoch in range(number_of_epochs): 
        graph_model.train() 
        total_loss = 0.0 

        for X, Y, Z, y in batch_training:    
            device = next(graph_model.parameters()).device
            y = y.to(device) 
            optimizing_factor.zero_grad() 

            node_features_batch, adjacency_batch = Z
            node_features_batch = node_features_batch.to(device)
            adjacency_batch = adjacency_batch.to(device)

            with autocast('cuda'):
                logits = graph_model(node_features_batch, adjacency_batch)
                batch_loss = loss_function(logits, y)
            scaler.scale(batch_loss).backwards()

            grads_ok = True 
            for p in graph_model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grads_ok = False 
                    break 

            if not grads_ok:
                optimizing_factor.zero_grad() 
            else:
                torch.nn.utils.clip_grad_norm_(graph_model.parameters(), max_norm=1.0) 
                scaler.step(optimizing_factor)
                scaler.update()

            total_loss = total_loss + batch_loss.item() 

        accuracy = model_accuracy_gnn(batch_training) 
        print(f"Epoch {epoch+1}/{number_of_epochs} - Loss: {total_loss:.4f} - Accuracy: {accuracy:.4f}") 

        torch.save({
            "epoch": epoch,
            "model": graph_model.state_dict(),
            "optimizer": optimizing_factor.state_dict(),
            "scaler": scaler.state_dict()
        }, "checkpoint.pt")

        s3.upload_file(
            "checkpoint.pt",
            S3_BUCKET,
            f"checkpoints/chess_convnext_epoch_{epoch}.pt"
        )