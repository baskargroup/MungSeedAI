# MungSeedAI

Companion code for:

> **Integrative Genomics and Deep Learning-Based Phenotyping Reveal the Genetic Architecture of Seed Traits in Mung bean (*Vigna radiata* L.)**
> Boddepalli, Jubery, Cannon, Farmer, Dutta, Ganapathysubramaniam, and Singh (2026)

Uses [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) to segment seeds from flatbed scanner images and extract per-seed morphological and color traits (area, axis lengths, aspect ratio, mean hue). Extracted traits feed downstream GWAS and genomic analyses in the companion manuscript.

---

## Repository Structure

```
MungSeedAI/
├── sam_for_seed.py       # Main script
├── requirements.txt
├── LICENSE
└── examples/
    ├── input/            # Example scanner image
    └── output/           # Example outputs
```

---

## Installation

```bash
conda create -n mungseed python=3.10 -y && conda activate mungseed
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Download the SAM ViT-L checkpoint and place it alongside the script:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

Other checkpoints (`vit_h`, `vit_b`) are available at the [SAM model zoo](https://github.com/facebookresearch/segment-anything#model-checkpoints). Update `sam_checkpoint` and `model_type` in the script accordingly.

---

## Usage

Edit the three variables at the top of `sam_for_seed.py`:

| Variable | Description |
|---|---|
| `input_directory_1` | Directory of input `.jpg`/`.png` scan images |
| `output_directory` | Directory for output files |
| `sam_checkpoint` | Path to downloaded SAM `.pth` file |

Then run:

```bash
python sam_for_seed.py
```

**Outputs per image:**

| File | Description |
|---|---|
| `<name>_color_mask.png` | SAM masks with random colors |
| `<name>_filtered_mask.png` | IQR-filtered masks applied to original image |
| `<name>_seed_properties.csv` | Per-seed traits: area, axis lengths, aspect ratio, mean hue |

**Hardware:** GPU strongly recommended (≥ 8 GB VRAM for ViT-L). Falls back to CPU automatically.

---

## Example

| Input | Color mask | Filtered mask |
|:---:|:---:|:---:|
| ![](examples/input/Green%20(PI%20201869)_1.jpg) | ![](examples/output/Green%20(PI%20201869)_1_color_mask.png) | ![](examples/output/Green%20(PI%20201869)_1_filtered_mask.png) |

---

## Citation

```bibtex
@article{boddepalli2026mungseedai,
  title  = {Integrative Genomics and Deep Learning-Based Phenotyping Reveal the Genetic Architecture of Seed Traits in Mung bean (\textit{Vigna radiata} L.)},
  author = {Boddepalli, Venkata Naresh and Jubery, Talukder Zaki and Cannon, Steven B. and Farmer, Andrew and Dutta, Somak and Ganapathysubramaniam, Baskar and Singh, Arti},
  year   = {2026}
}
```

Please also cite SAM: Kirillov et al., *Segment Anything*, arXiv:2304.02643 (2023).

---

## License

MIT — see [LICENSE](LICENSE).

## Contact

Code: Zaki Jubery (znjubery@iastate.edu) and Baskar Ganapahtysubramanian (baskarg@iastate.edu) · Manuscript: Arti Singh (artisingh@iastate.edu)
