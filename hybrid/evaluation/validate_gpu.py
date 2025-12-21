import torch
import subprocess
import sys

def validate_gpu():
    """
    Validates GPU availability and provides detailed diagnostics.
    """
    print("--- GPU Validation ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if not cuda_available:
        print("\nCUDA not available. Collecting diagnostics...")
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("\nnvidia-smi output:")
                print(result.stdout)
            else:
                print(f"\nnvidia-smi failed with exit code {result.returncode}:")
                print(result.stderr)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"\nCould not run nvidia-smi: {e}")

        sys.exit(1)

    print(f"\nFound {torch.cuda.device_count()} CUDA device(s).")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

    print("\n✓ GPU validation successful.")

if __name__ == "__main__":
    validate_gpu()
