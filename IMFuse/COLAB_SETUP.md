# Google Colab Setup Instructions for IMFuse

## Quick Start for Google Colab

### 1. Clone the Repository
```bash
!git clone https://github.com/AImageLab-zip/IM-Fuse.git
%cd IM-Fuse/IMFuse
```

### 2. Install PyTorch with CUDA Support (MUST BE FIRST)
```bash
# Install PyTorch from PyTorch's index (not PyPI)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install All Other Dependencies
```bash
# Now install everything else from requirements.txt
!pip install -r requirements.txt
```

### 4. Verify Installation
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

# Test IMFuse model import
from IMFuse_no1skip import Model
print("✓ IMFuse model loaded successfully")
```

---

## Complete Colab Workflow

### Step 1: Mount Google Drive (for data and checkpoints)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Setup Environment
```bash
# Clone and install
!git clone https://github.com/AImageLab-zip/IM-Fuse.git
%cd IM-Fuse/IMFuse

# Install PyTorch FIRST
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
!pip install -r requirements.txt
```

### Step 3: Preprocess Data
Edit `preprocess.py` with your Google Drive paths:
```python
src_path = '/content/drive/MyDrive/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
tar_path = '/content/drive/MyDrive/BraTS2023/BRATS2023_Training_npy'
```

Then run:
```bash
!python preprocess.py
```

### Step 4: Train IMFuse
```bash
!python train_poly.py \
  --datapath /content/drive/MyDrive/BraTS2023/BRATS2023_Training_npy \
  --dataname BRATS2023 \
  --savepath /content/drive/MyDrive/BraTS2023/imfuse_runs/exp1 \
  --num_epochs 1000 \
  --batch_size 1 \
  --mamba_skip \
  --interleaved_tokenization
```

### Step 5: Test/Evaluate
```bash
!python test.py \
  --datapath /content/drive/MyDrive/BraTS2023/BRATS2023_Training_npy \
  --dataname BRATS2023 \
  --savepath /content/drive/MyDrive/BraTS2023/imfuse_results/exp1_test \
  --resume /content/drive/MyDrive/BraTS2023/imfuse_runs/exp1/model_best.pth.tar \
  --batch_size 2 \
  --mamba_skip \
  --interleaved_tokenization
```

---

## Troubleshooting

### Issue: Dependency conflicts
These warnings are normal in Colab and can usually be ignored. IMFuse uses specific package versions that may conflict with Colab's pre-installed packages, but this doesn't prevent IMFuse from running.

### Issue: Out of memory
- Reduce `--batch_size` to 1
- Use gradient checkpointing (if implemented)
- Restart runtime and clear cache: `Runtime → Restart runtime`

### Issue: Disconnected from runtime
- Enable browser notifications for Colab
- Consider Colab Pro for longer runtimes
- Save checkpoints frequently (model saves automatically every epoch)

---

## Notes
- **Always install PyTorch BEFORE requirements.txt**
- Colab GPU sessions have time limits (varies by plan)
- Save all outputs to Google Drive to persist after session ends
- Training 1000 epochs may require multiple sessions - use `--resume` to continue

