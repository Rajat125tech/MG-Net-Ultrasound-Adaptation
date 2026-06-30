# MG-Net 2D: Fetal Ultrasound Segmentation

A faithful, high-performance 2D adaptation of the **MG-Net** architecture for prenatal fetal ultrasound segmentation. This repository adapts the original 3D fetal MRI segmentation network to process 2D ultrasound images and segment key brain structures: **Brain, Cavum Septi Pellucidi (CSP), and Lateral Ventricles (LV)**.

---

## 🔬 Motivation
The original MG-Net was designed to perform segmentation on 3D NIfTI volumes from fetal MRI scans. Fetal ultrasound scans, however, are typically acquired as 2D PNG slices, which present unique challenge profiles:
* High levels of speckle noise and acoustic shadows.
* Severe class imbalance (fetal brain structures occupy small regions).
* Faster clinical workflows requiring real-time, lightweight inference.

This repository converts the entire MG-Net framework—including the Swin/UX-Net hybrid encoder, Grapher Bottlenecks, Spatial Attention Decoders, and Multi-Scale Deformable Attention skip connections—to **2D space**, offering a robust deep learning pipeline for prenatal ultrasound head plane measurements.

---

## 🤝 Acknowledgement
This work is based on the original MG-Net repository:
* **Original Repository:** [unicoco7/MG-Net](https://github.com/unicoco7/MG-Net)
* **Original Domain:** 3D fetal brain MRI segmentation on the FeTA dataset.

---

## 🔄 Summary of 3D MRI ➡️ 2D Ultrasound Conversion

The adaptation preserves the original MG-Net architecture, training strategy, and logic while mapping all spatial and tensor operations from 3D to 2D equivalents:
1. **Dimension Slicing (5D ➡️ 4D):** All model blocks now receive 4D tensors `(Batch, Channel, Height, Width)` instead of 5D `(Batch, Channel, Height, Width, Depth)`.
2. **Convolutional & Normalization Layers:** Swapped `Conv3d`, `MaxPool3d`, `ConvTranspose3d`, `BatchNorm3d`, and `InstanceNorm3d` with their corresponding 2D operators.
3. **Neighborhood Attention (NAT3D ➡️ NAT2D):** Swapped 3D Neighborhood Attention (`NeighborhoodAttention3D`) and 3D Deformable Convolution (`DCNv3_3d`) with their 2D counterparts (`NeighborhoodAttention2D` and `DCNv3_pytorch`). Added a pure-PyTorch fallback module for environments without compiled NATTEN C++ extensions.
4. **Grapher Bottlenecks (GCB):** Replaced the 3D Grapher block (`GCB3D`) with the 2D version (`GCB`). Re-calculated the bottleneck Grapher node count $n$ from $216$ ($6 \times 6 \times 6$) to $36$ ($6 \times 6$), matching the downsampled resolution.
5. **Pixel Decoder Skip Connections:** Converted the multi-scale deformable attention decoder (`MSDeformAttnPixelDecoder3D`) to a 2D multi-scale attention module (`MSDeformAttnPixelDecoder2D`), adapting reference coordinates and attention grid normalization to 2D.
6. **Ultrasound Data Pipeline:** Replaced MRI-specific windowing and padding with a custom `LoadPNGImaged` loader, scaling, and 2D spatial transformations.

---

## 📊 Dataset Description
The model is configured to segment prenatal fetal ultrasound scan images with PASCAL VOC-style segmentation masks. 

### Class Distribution (Inspected across 999 scans)
| Class Index | Structure / Region | RGB Color Map | Pixel Count | % of Dataset | Occurrence in Dataset |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | **Background** | `[0, 0, 0]` | 454,611,752 | 71.79% | 999 / 999 files |
| **1** | **Brain** | `[255, 0, 0]` (Red) | 175,647,905 | 27.74% | 999 / 999 files |
| **2** | **CSP** (Cavum Septi Pellucidi) | `[255, 255, 0]` (Yellow) | 1,110,272 | 0.18% | 408 / 999 files |
| **3** | **LV** (Lateral Ventricles) | `[0, 0, 255]` (Blue) | 1,895,172 | 0.30% | 510 / 999 files |

---

## 📁 Repository Structure
```
MG-Net/
├── MG-Net/
│   ├── load_datasets_transforms.py   # Custom 2D PNG loading, RGB color mapping, & splitting
│   ├── main_train.py                 # 2D Training and Validation loops
│   └── networks/
│       └── MG_Net_3D/
│           ├── network_backbone.py   # MG_Net_2D architecture main definition
│           ├── uxnet_encoder.py      # 2D ConvNeXt-style encoder with NAT2D/DCNv3
│           ├── skipconnect3d.py      # 2D Deformable Attention Pixel Decoder
│           ├── decoders.py           # Spatial Attention (SPA) module adapted to 2D
│           ├── CGCN_UpBlock.py       # Dynamic MONAI-based 2D/3D decoder blocks
│           ├── swin_transformer.py   # Swin ViT adapter supporting 2D configurations
│           └── gcn_lib/
│               └── pos_embed.py      # Positional embeddings (updated to NumPy 1.24+)
├── requirements.txt                  # Core project requirements
└── .gitignore                        # Python gitignore configuration
```

---

## ⚙️ Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/unicoco7/MG-Net.git
   cd MG-Net
   ```

2. **Set up Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🗂️ Expected Folder Structure
For dataset ingestion, structure your directories as relative paths under your dataset root directory:

```
dataset_root/
├── Orginal_train_images_to_959_661/
│   ├── 000_HC.png
│   ├── 001_HC.png
│   └── ...
└── Mask/
    └── Test-Dataset-Segmentation/
        └── SegmentationClass/
            ├── 000_HC.png
            ├── 001_HC.png
            └── ...
```

---

## 🚀 Execution Instructions

### 1. Model Training
To train the 2D model on the ultrasound dataset, run:
```bash
python MG-Net/main_train.py \
    --root "./dataset_root" \
    --output "./output" \
    --dataset "ultrasound" \
    --network "MG_Net_2D" \
    --max_iter 10000 \
    --eval_step 100 \
    --lr 1e-4
```
*Note: The dataset pipeline deterministically splits the images `80/20` into training and validation folds.*

### 2. Model Evaluation
To run validation sliding-window inference and output predictions:
```bash
python MG-Net/main_train.py \
    --root "./dataset_root" \
    --output "./output" \
    --dataset "ultrasound" \
    --network "MG_Net_2D" \
    --mode "test" \
    --pretrain True \
    --pretrained_weights "./output/best_metric_model.pth"
```

---

## ⚠️ Current Limitations & Future Work

* **Class Imbalance:** Brain structures like CSP and LV constitute less than `0.5%` of the total pixel volume. In future iterations, implementing custom class weights in the `DiceCELoss` or focal loss schemes could improve boundary segmentation for these small regions.
* **Hardware Requirements:** Although the PyTorch Neighborhood Attention fallback runs successfully on CPU, full training should be conducted on CUDA-enabled GPUs for viable processing speeds.
