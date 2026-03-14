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

MungSeedAI uses Meta AI's [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) to automatically segment individual seeds from flatbed scanner images. Seeds are arranged in a 6 × 10 physical grid; the pipeline:

1. Runs SAM automatic mask generation on each cropped scan image
2. Filters masks by area (IQR-based outlier removal) to retain only seed-sized objects
3. Measures morphological traits per seed: area, major/minor axis length, aspect ratio, and mean hue (HSV)
4. Assigns each detected seed to its numbered grid position using a bipartite nearest-neighbor matching
5. Saves per-image outputs: color mask, filtered mask, annotated object-ID image, transparent contour overlay, and a CSV of seed properties

The trait data feed downstream GWAS and genomic analyses reported in the companion manuscript.

---

## Repository Structure

```
MungSeedAI/
├── sam_for_seed_gt2026.py   # Main segmentation and trait extraction script
├── requirements.txt         # Python dependencies
├── LICENSE
└── examples/
    ├── input/
    │   └── Green (PI 201869)_1.jpg          # Example scanner image (cropped, 1 accession)
    └── output/
        ├── Green (PI 201869)_1_color_mask.png        # Random-color SAM masks
        ├── Green (PI 201869)_1_filtered_mask.png     # Area-filtered binary mask
        ├── Green (PI 201869)_1_object_ids.png        # Annotated seed IDs
        ├── Green (PI 201869)_1_transparent_overlay.png  # Contour overlay on original
        └── Green (PI 201869)_1_seed_properties.csv  # Per-seed trait table
```

---

## Example Output

| Input scan | Object IDs | Transparent overlay |
|:---:|:---:|:---:|
| ![input](examples/input/Green%20(PI%20201869)_1.jpg) | ![ids](examples/output/Green%20(PI%20201869)_1_object_ids.png) | ![overlay](examples/output/Green%20(PI%20201869)_1_transparent_overlay.png) |

Color-coded SAM masks and filtered mask:

| Color mask | Filtered mask |
|:---:|:---:|
| ![color](examples/output/Green%20(PI%20201869)_1_color_mask.png) | ![filtered](examples/output/Green%20(PI%20201869)_1_filtered_mask.png) |

Example CSV output (`Green (PI 201869)_1_seed_properties.csv`):

| Object Number | Mask Index | Area | Major Axis Length | Minor Axis Length | Aspect Ratio | Mean Hue | Centroid Row | Centroid Col | Grid Number | Grid Row | Grid Col | Distance To Grid |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | ... | ... | ... | ... | ... | ... | ... | ... | 1 | 1 | 1 | ... |

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/<your-org>/MungSeedAI.git
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

```bash
# Run on the provided example (output written to seed_gt_cropped_grid/)
python sam_for_seed_gt2026.py
```

By default the script reads from `../data/seed_gt/cropped` and writes outputs to `seed_gt_cropped_grid/`. To run on the bundled example, edit the path variables at the top of the script:

```python
input_directory_1 = "examples/input"
output_directory  = "examples/output_run"
```

### Configuration

All key parameters are defined as constants near the top of `sam_for_seed_gt2026.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sam_checkpoint` | `"sam_vit_l_0b3195.pth"` | Path to the SAM model weights |
| `model_type` | `"vit_l"` | SAM backbone (`vit_h`, `vit_l`, `vit_b`) |
| `GRID_ROWS` | `6` | Number of rows in the seed tray |
| `GRID_COLS` | `10` | Number of columns in the seed tray |
| `MAX_DISTANCE_TO_GRID` | `60.0` | Maximum pixel distance to assign a seed to a grid cell |
| `TOP_LEFT` | `(209, 316)` | Top-left corner of the seed grid (pixels) |
| `TOP_RIGHT` | `(910, 306)` | Top-right corner |
| `BOTTOM_LEFT` | `(199, 705)` | Bottom-left corner |
| `BOTTOM_RIGHT` | `(902, 696)` | Bottom-right corner |

> **Note:** The corner coordinates are image-specific. Adjust them to match your scan setup before running on new images. A 6 × 10 grid gives 60 seed positions numbered left-to-right, top-to-bottom.

### Output files per image

For each input image `<name>.jpg` the following files are written to the output directory:

| File | Description |
|------|-------------|
| `<name>_color_mask.png` | SAM masks rendered with random colors |
| `<name>_filtered_mask.png` | Masks after IQR area filtering, applied to the original image |
| `<name>_object_ids.png` | Original image annotated with seed centroids and grid-assigned IDs |
| `<name>_transparent_overlay.png` | Original image with white contour outlines of filtered seeds |
| `<name>_seed_properties.csv` | Per-seed table: area, axis lengths, aspect ratio, mean hue, grid position |

---

## Hardware Requirements

SAM inference is GPU-accelerated. The script automatically falls back to CPU if no CUDA device is detected, but GPU is strongly recommended for reasonable throughput.

- **GPU (recommended):** NVIDIA GPU with ≥ 8 GB VRAM (ViT-L); ≥ 16 GB for ViT-H
- **CPU (fallback):** Processing will be significantly slower (~minutes per image)
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
