import torch

def main():
    print("Hello from inference-lab!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Metal Performance Shaders available: {torch.backends.mps.is_available()}")


if __name__ == "__main__":
    main()
