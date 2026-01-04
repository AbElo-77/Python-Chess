import sys
import os
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.algorithmic_processing.models.neural_network_models.convolutional_nn_model import ChessConvNeXt
from backend.algorithmic_processing.models.neural_network_models.transformer_model import ChessDaViT
from backend.algorithmic_processing.models.neural_network_models.graph_nn_model import ChessSTGNN


class TestChessConvNeXt:
    
    def test_forward_pass(self):
        batch_size = 4
        model = ChessConvNeXt(num_classes=100)

        X = torch.randn(batch_size, 12, 8, 8)
        
        logits = model(X)
        
        assert logits.shape == (batch_size, 100), f"Expected shape ({batch_size}, 100), got {logits.shape}"
    
    def test_device_compatibility(self):
        device = torch.device("cpu")
        model = ChessConvNeXt(num_classes=50).to(device)
        
        X = torch.randn(2, 12, 8, 8, device=device)
        logits = model(X)
        
        assert logits.device.type == "cpu"
        assert logits.shape == (2, 50)


class TestChessDaViT:
    
    def test_forward_pass(self):
        batch_size = 4
        model = ChessDaViT(num_classes=100)

        X = torch.randn(batch_size, 12, 8, 8)
        
        logits = model(X)
        
        assert logits.shape == (batch_size, 100), f"Expected shape ({batch_size}, 100), got {logits.shape}"
    
    def test_device_compatibility(self):
        device = torch.device("cpu")
        model = ChessDaViT(num_classes=50).to(device)
        
        X = torch.randn(2, 12, 8, 8, device=device)
        logits = model(X)
        
        assert logits.device.type == "cpu"
        assert logits.shape == (2, 50)


class TestChessSTGNN:
    
    def test_forward_pass_unbatched(self):
        model = ChessSTGNN(
            in_channels=12,
            hidden_channels=64,
            out_channels=64,
            num_heads=4,
            embed_dims=64,
            num_classes=100
        )
        
        X = torch.randn(64, 12)
        A = torch.randn(64, 64)
        
        logits = model(X, A)
        
        assert logits.shape == (1, 100), f"Expected shape (1, 100), got {logits.shape}"
    
    def test_forward_pass_batched(self):
        batch_size = 4
        model = ChessSTGNN(
            in_channels=12,
            hidden_channels=64,
            out_channels=64,
            num_heads=4,
            embed_dims=64,
            num_classes=100
        )
        
        X = torch.randn(batch_size, 64, 1, 12)  
        A = torch.randn(batch_size, 64, 64)     
        
        logits = model(X, A)
        
        assert logits.shape == (batch_size, 100), f"Expected shape ({batch_size}, 100), got {logits.shape}"
    
    def test_encode_method(self):
        model = ChessSTGNN(
            in_channels=12,
            hidden_channels=64,
            out_channels=64,
            num_heads=4,
            embed_dims=64,
            num_classes=100
        )
        
        X = torch.randn(64, 12)
        A = torch.randn(64, 64)
        
        embeddings = model.encode(X, A)
        
        assert embeddings.shape == (1, 64), f"Expected shape (1, 64), got {embeddings.shape}"
    
    def test_device_compatibility(self):
        device = torch.device("cpu")
        model = ChessSTGNN(
            in_channels=12,
            hidden_channels=64,
            out_channels=64,
            num_heads=4,
            embed_dims=64,
            num_classes=50
        ).to(device)
        
        X = torch.randn(64, 12, device=device)
        A = torch.randn(64, 64, device=device)
        
        logits = model(X, A)
        
        assert logits.device.type == "cpu"
        assert logits.shape == (1, 50)


class TestDataLoader:
    
    def test_safe_loader_finite_iteration(self):
        from backend.algorithmic_processing.pre_post_processing.input_to_tensor import create_loader, generate_moves_made
        
        test_files = ['./data/training_dataset/page_1.csv']
        
        try:

            move_to_id, _ = generate_moves_made(test_files)
            loader = create_loader(batch_size=4, input_files=test_files, move_to_id=move_to_id)
    
            max_iterations = len(loader)
            iteration_count = 0
            for batch in loader:
                iteration_count += 1
                if iteration_count > max_iterations + 5:  
                    raise RuntimeError(f"DataLoader exceeded max iterations: {iteration_count} > {max_iterations + 5}")
            
            assert iteration_count == max_iterations or iteration_count <= (len(loader.dataset) + 3) // 4
        except FileNotFoundError:
            print("General Error")


def run_all_tests():
    
    tests_passed = 0
    tests_failed = 0
    
    test_classes = [
        TestChessConvNeXt,
        TestChessDaViT,
        TestChessSTGNN,
        TestDataLoader,
    ]
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 40)
        
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        
        for method_name in methods:
            try:
                method = getattr(instance, method_name)
                method()
                tests_passed += 1
            except Exception as e:
                tests_failed += 1
        
        print(f"Tests Passed: {tests_passed}")
        print(f"Tests Failed: {tests_failed}")

    
    return tests_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
