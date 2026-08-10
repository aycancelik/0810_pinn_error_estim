import torch


def set_default_device(device: str = "cpu"):
    if device == "cpu":
        torch.set_default_device("cpu")
        print("Using CPU")
    elif device == "cuda":
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
            print("Using CUDA")
        else:
            print("CUDA not available. Using CPU")
            torch.set_default_device("cpu")
    elif device == "mps":
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            torch._dynamo.disable()
            torch.set_default_device("mps")
            torch._dynamo.reset()
            print("Using mps")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            print("Using MPS")
        else:
            print("MPS not available. Using CPU")
            torch.set_default_device("cpu")
    else:
        print("Invalid device. Using CPU")
        torch.set_default_device("cpu")
