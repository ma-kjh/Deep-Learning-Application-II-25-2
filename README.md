
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

