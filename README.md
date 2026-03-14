# MungSeedAI

**SAM-based automated seed phenotyping for mung bean (*Vigna radiata* L.)**

This repository contains the image segmentation and seed trait extraction code accompanying the manuscript:

> **Integrative Genomics and Deep Learning-Based Phenotyping Reveal the Genetic Architecture of Seed Traits in Mung bean (*Vigna radiata* L.)**
>
> Venkata Naresh Boddepalli, Talukder Zaki Jubery, Steven B. Cannon, Andrew Farmer, Somak Dutta, Baskar Ganapathysubramaniam, and Arti Singh
>
> *Manuscript in preparation / under review, 2026*

---

## Overview

MungSeedAI uses Meta AI's [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) to automatically segment individual seeds from flatbed scanner images and extract per-seed morphological and color traits. The pipeline:

1. Runs SAM automatic mask generation on each input image
2. Filters masks by area using IQR-based outlier removal to retain only seed-sized objects
3. Measures morphological traits per seed: area, major/minor axis length, aspect ratio, and mean hue (HSV)
4. Saves per-image outputs: a random-color mask, a filtered binary mask applied to the original image, and a CSV of seed properties

The extracted traits feed downstream GWAS and genomic analyses reported in the companion manuscript.

---

## Repository Structure

```
MungSeedAI/
├── sam_for_seed.py          # Main segmentation and trait extraction script
├── requirements.txt         # Python dependencies
├── LICENSE
└── examples/
    ├── input/
    │   └── Green (PI 201869)_1.jpg          # Example scanner image (cropped, 1 accession)
    └── output/
        ├── Green (PI 201869)_1_color_mask.png        # Random-color SAM masks
        ├── Green (PI 201869)_1_filtered_mask.png     # Area-filtered mask applied to original
        └── Green (PI 201869)_1_seed_properties.csv  # Per-seed trait table
```

---

## Example Output

| Input scan | Color mask | Filtered mask |
|:---:|:---:|:---:|
| ![input](examples/input/Green%20(PI%20201869)_1.jpg) | ![color](examples/output/Green%20(PI%20201869)_1_color_mask.png) | ![filtered](examples/output/Green%20(PI%20201869)_1_filtered_mask.png) |

Example CSV output (`Green (PI 201869)_1_seed_properties.csv`):

| Area | Major Axis Length | Minor Axis Length | Aspect Ratio | Mean Hue |
|:---:|:---:|:---:|:---:|:---:|
| 4832 | 89.3 | 67.1 | 1.33 | 42.7 |
| ... | ... | ... | ... | ... |

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/znjubery/MungSeedAI.git
cd MungSeedAI
```

### 2. Create a conda environment (recommended)

```bash
conda create -n mungseed python=3.10 -y
conda activate mungseed
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Segment Anything Model (SAM)

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 5. Download the SAM checkpoint

The script uses the **ViT-L** SAM model. Download the checkpoint and place it in the same directory as the script (or update the path in the script):

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

Alternative checkpoints (update `sam_checkpoint` and `model_type` in the script accordingly):

| Model | Checkpoint file | `model_type` |
|-------|----------------|--------------|
| ViT-H (largest) | `sam_vit_h_4b8939.pth` | `"vit_h"` |
| ViT-L (used here) | `sam_vit_l_0b3195.pth` | `"vit_l"` |
| ViT-B (smallest) | `sam_vit_b_01ec64.pth` | `"vit_b"` |

All checkpoints are available at the [SAM model zoo](https://github.com/facebookresearch/segment-anything#model-checkpoints).

---

## Usage

### Quick start with the example image

Edit the path variables at the top of `sam_for_seed.py`:

```python
input_directory_1 = "examples/input"
output_directory  = "examples/output_run"
```

Then run:

```bash
python sam_for_seed.py
```

### Configuration

Key parameters at the top of `sam_for_seed.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_directory_1` | `"../processed_data/..."` | Path to directory of input images |
| `output_directory` | `"..."` | Path for output files |
| `sam_checkpoint` | `"sam_vit_l_0b3195.pth"` | Path to SAM model weights |
| `model_type` | `"vit_l"` | SAM backbone (`vit_h`, `vit_l`, `vit_b`) |

Area filtering uses a standard IQR filter (Q1 − 1.5×IQR, Q3 + 1.5×IQR) to automatically remove background detections and outlier-sized objects.

### Output files per image

For each input image `<name>.jpg` the following files are written to the output directory:

| File | Description |
|------|-------------|
| `<name>_color_mask.png` | SAM masks rendered with random colors |
| `<name>_filtered_mask.png` | Area-filtered masks applied to the original image |
| `<name>_seed_properties.csv` | Per-seed table: area, axis lengths, aspect ratio, mean hue |

---

## Hardware Requirements

SAM inference is GPU-accelerated. The script automatically falls back to CPU if no CUDA device is detected, but GPU is strongly recommended.

- **GPU (recommended):** NVIDIA GPU with ≥ 8 GB VRAM (ViT-L); ≥ 16 GB for ViT-H
- **CPU (fallback):** Significantly slower (~minutes per image)
- **Tested on:** NVIDIA A100 80 GB (HPC cluster)

---

## Citation

If you use this code, please cite:

```bibtex
@article{boddepalli2026mungseedai,
  title   = {Integrative Genomics and Deep Learning-Based Phenotyping Reveal the Genetic Architecture of Seed Traits in Mung bean (\textit{Vigna radiata} L.)},
  author  = {Boddepalli, Venkata Naresh and Jubery, Talukder Zaki and Cannon, Steven B. and Farmer, Andrew and Dutta, Somak and Ganapathysubramaniam, Baskar and Singh, Arti},
  journal = {(under review)},
  year    = {2026}
}
```

Please also cite the Segment Anything Model:

```bibtex
@article{kirillov2023segment,
  title   = {Segment Anything},
  author  = {Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal = {arXiv:2304.02643},
  year    = {2023}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions about the code, contact **Talukder Zaki Jubery** (jubery@iastate.edu) or open an issue on GitHub.
For questions about the manuscript or phenotyping pipeline, contact **Arti Singh** (artisingh@iastate.edu).
