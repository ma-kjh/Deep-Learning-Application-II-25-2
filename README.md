
# [Project] Re-evaluating OoD Detection for Sigmoid-based Vision-Language Models (SigLIP)

---

## 🔧 Experimental Environment

We ensure high reproducibility by specifying the exact hardware and software configurations used in our experiments.

### Hardware Infrastructure
* **GPU:** NVIDIA RTX 6000 **(Blackwell Architecture)**
* **Driver/CUDA:** 535.104.05 / CUDA 12.8

### Software Dependencies
Ensure the following dependencies are installed:

```bash
conda create -n siglip_ood python=3.11.14
conda activate siglip_ood
pip install -r requirements.txt
````

**Core Libraries:**

  * **Python**: 3.11.14
  * **PyTorch**: 2.9.1+cu128
  * **Torchvision**: 0.24.1+cu128
  * **Numpy**: 2.3.5
  * **Scikit-learn**: 1.7.2
  * **OpenCLIP**: 2.24.0

-----

## 📂 Dataset Preparation

### In-Distribution (ID)

  * **Dataset:** ImageNet-1K (ILSVRC 2012)
  * Structure:
    ```
    /data/imagenet/
    ├── train/
    └── val/
    ```

### Out-of-Distribution (OoD)

We utilize the standard OoD benchmark datasets defined in **MOS**.

  * **Datasets:** iNaturalist, SUN, Places, Textures

-----

## 🚀 Quickstart

You can reproduce our results using the provided `main.py`.

### 1\. Baseline: CLIP (Softmax MCM)

Reproducing the standard CLIP performance using the original MCM method.

```bash
python main.py \
    --model_type clip \
    --ood-dataset iNaturalist \
    --bs 1024
```

### 2\. The Problem: SigLIP with Naive Softmax

Running SigLIP with the traditional Softmax-based MCM. (Performance is suboptimal due to the "Bias Erasure" issue).

```bash
python main.py \
    --model_type siglip2 \
    --ood-dataset iNaturalist \
    --bs 1024
    # Note: This runs Softmax logic by default for comparison
```

### 3\. **The Solution: SigLIP with Adaptive Sigmoid (Ours)**

Running our proposed method. This utilizes **Sigmoid MCM**, preserving the learned `logit_bias` and absolute confidence scales.

```bash
# The code automatically detects 'siglip' model_type and applies Sigmoid logic
python main.py \
    --model_type siglip2 \
    --ood-dataset iNaturalist \
    --bs 1024 \
    --use_sigmoid  # (Optional flag if you implemented argument control)
```

-----

## 📊 Results & Analysis

We compared the OoD detection performance (AUROC) between the Naive Softmax approach and our Proposed Sigmoid approach on SigLIP-2.

| Model | Inference Method | ImageNet ACC | OOD AUROC (Avg) |
|:---:|:---:|:---:|:---:|
| **CLIP** (ViT-B-16) | Standard MCM (Softmax) | 68.3% | 85.2% |
| **SigLIP** (ViT-B-16) | Naive MCM (Softmax) | 75.1% | 86.5% |
| **SigLIP** (ViT-B-16) | **Adaptive MCM (Sigmoid)** | **75.1%** | **89.8%** |

### 💡 Key Finding: Why Sigmoid?

  * **Bias Preservation:** SigLIP learns a scalar `logit_bias` to shift negative logits. Softmax is translation invariant ($Softmax(x+c) = Softmax(x)$), effectively **erasing** this learned rejection threshold. Sigmoid respects the bias.
  * **Independence:** Softmax forces the sum of probabilities to be 1, causing the **"Forced Winner"** problem on OoD data. Sigmoid allows all class probabilities to be near 0, correctly identifying OoD samples.
