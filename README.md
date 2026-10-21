# GUDA: Unsupervised Domain Adaptation in Biomedical Images Segmentation With Guided Diffusion Generative Prior

Official PyTorch implementation of the paper
[*"Unsupervised Domain Adaptation in Biomedical Images Segmentation With Guided Diffusion Generative Prior"*](https://doi.org/10.1109/TIP.2026.3678763),
IEEE Transactions on Image Processing, vol. 35, pp. 3690–3704, 2026.

Alexandre Stenger, Étienne Baudrier, Nicolas Passat, Benoît Naegel

GUDA casts segmentation as conditional generation: a denoising diffusion model learns the distribution of the source
masks, guided by the image and by a pseudo-label from Segment Anything (SAM). Under the covariate shift assumption
(same mask distribution in both domains, P_s(Y|X) = P_t(Y|X)), this learned distribution acts as a generative prior
when segmenting the target domain. Adaptation is done by self-training on the target SAM pseudo-labels, while keeping
the supervised loss on the source domain. No target label is used.

## Method in a nutshell

1. **SAM pseudo-labels** ([tools/sam_pseudolabels.py](tools/sam_pseudolabels.py)): every image (source and target)
   is segmented offline by SAM (ViT-H) in automatic "segment everything" mode, prompted by a regular 32 × 32 point
   grid. All the shapes found are merged into one binary pseudo-label, except the masks that are too large (SAM often
   returns a mask covering the whole image).
2. **Guided diffusion**: the denoising network is an attention U-Net (128 channels in the first layer, 4 levels in
   the encoder and in the decoder, self-attention with 1 head at resolution 16). It takes as input, concatenated
   along channels, the noisy mask, the image and its SAM pseudo-label, and predicts the noise on the mask:

   ```
   [ noisy mask (3) | image (3) | SAM pseudo-label (3) ]  ->  predicted noise (3)
   ```

3. **Training** ([tools/train.py](tools/train.py)), with the source loss on the real source labels and the target
   self-training loss on the target SAM pseudo-labels:

   ```
   L_global = L_source + lambda * L_target
   L_source = || eps - eps_theta(Y_t^s,  X^s, pi^s, t) ||^2      (noise added to the source label Y^s)
   L_target = || eps - eps_theta(pi_t^t, X^t, pi^t, t) ||^2      (noise added to the target pseudo-label pi^t)
   ```

   The model is first warmed up on the source domain only (`--stage source`, lambda = 0, 200,000 iterations), then
   trained on both domains (`--stage adapt`, lambda = 1e-4, 800,000 iterations). The final model is used.
4. **Inference** ([tools/generate.py](tools/generate.py), [tools/evaluate.py](tools/evaluate.py)): N = 10 masks are
   sampled per target image with DDIM (100 sampling steps, 1,000 noising steps), conditioned on the image and its
   SAM pseudo-label. They are resized to the image size, averaged and thresholded at gamma = 0.5.

## Installation

```bash
git clone https://github.com/alex-stenger/GUDA.git
cd GUDA
pip install -r requirements.txt   # install PyTorch first if you need a specific CUDA build
```

Training requires a CUDA GPU. The code was tested with Python 3.10 and PyTorch 2.5. To compute the SAM pseudo-labels,
also install Segment Anything and download the ViT-H weights:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Data

### Datasets of the paper

| Task | Domains | Source |
| --- | --- | --- |
| Mitochondria | FS1, FS2, WeiH | [FS1](https://zenodo.org/records/8341172), [FS2](https://zenodo.org/records/8344292), [WeiH (EMPIAR-11037)](https://www.ebi.ac.uk/empiar/EMPIAR-11037) |
| Endoplasmic reticulum | FS1, FS2 | [FS1](https://zenodo.org/records/8341172), [FS2](https://zenodo.org/records/8344292) |
| Brain tumour | Flair, T2 | [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html) |

For each domain, 5,000 patches are used for training and 500 for testing (256 × 256 for the electron microscopy
datasets, 240 × 240 for BraTS).

### Folder structure

Each domain is a folder with a `train` and a `test` split, containing the images (`img`), the ground-truth masks
(`lbl`) and the SAM pseudo-labels (`pred_sam`):

```
datasets/
└── <domain>/
    ├── train/
    │   ├── img/<any_subfolder>/case_001.png
    │   ├── lbl/<any_subfolder>/case_001.png        (source domain only)
    │   └── pred_sam/<any_subfolder>/case_001.png
    └── test/
        ├── img/...
        ├── lbl/...                                 (only used to compute the scores)
        └── pred_sam/...
```

- Files are matched by name (without extension) across `img`, `lbl` and `pred_sam`, and can be placed in nested
  sub-folders.
- Masks are binary images (background 0, foreground 255). Images are loaded as RGB.
- Everything is resized to `--image_size` (64 × 64 in the paper). The scores are computed at the resolution of the
  ground-truth masks.
- The target labels are never read during training.

### SAM pseudo-labels

```bash
python tools/sam_pseudolabels.py --dataset_path datasets/<domain> --checkpoint sam_vit_h_4b8939.pth
```

This writes `<split>/pred_sam/` for the `train` and `test` splits. Masks covering more than `--max_area_ratio` of
the image (default 0.5) are discarded; adjust it if the objects to segment are very large.

## Usage

The full pipeline for one adaptation scenario is in [scripts/run_pipeline.sh](scripts/run_pipeline.sh):

```bash
bash scripts/run_pipeline.sh datasets FS1 WeiH
```

The individual steps are as follows.

**1. Warm-up on the source domain**

```bash
python tools/train.py --stage source --run_name FS1_source \
  --dataset_source_path datasets/FS1 --result_path workdir
```

**2. Adaptation to the target domain**

```bash
python tools/train.py --stage adapt --run_name FS1-WeiH \
  --dataset_source_path datasets/FS1 --dataset_target_path datasets/WeiH \
  --pretrain_path workdir/FS1_source/last.pt --result_path workdir
```

Each run writes to `workdir/<run_name>/`:

| File | Content |
| --- | --- |
| `last.pt` | model at the end of the training (saved every `--save_interval` iterations), used by `--resume True` |
| `tensorboard/` | losses (`tensorboard --logdir workdir`) |
| `train.log` | arguments of the run |

**3. Sampling masks on the target test set**

```bash
python tools/generate.py --weight_path workdir/FS1-WeiH/last.pt \
  --dataset_path datasets/WeiH --split test --result_path results/FS1-WeiH/seed_0 --num_sample 10 --seed 0
```

This writes `<name>_<k>.png` for `k = 0..num_sample-1`.

**4. Evaluation**

```bash
python tools/evaluate.py --pred_path results/FS1-WeiH/seed_0 --gt_path datasets/WeiH/test/lbl --n_mask 10
```

The script reports the foreground IoU and Dice, and their mean over foreground and background. The averaged masks
(`<name>_mean.png`) and the thresholded masks (`<name>_mean_thresholded.png`) are saved next to the predictions.
Since predictions are sampled, the paper reports the mean and standard deviation over 10 seeds, as done in
[scripts/run_pipeline.sh](scripts/run_pipeline.sh).

### Main arguments of `tools/train.py`

| Argument | Default | Description |
| --- | --- | --- |
| `--stage` | required | `source` (warm-up) or `adapt` |
| `--dataset_source_path` / `--dataset_target_path` | | domain folders |
| `--pretrain_path` | | checkpoint to start from (the source model for `adapt`) |
| `--lambda_st` | `1e-4` | weight lambda of the target self-training loss |
| `--iterations` | `200000` (source), `800000` (adapt) | number of training iterations |
| `--batch_size` / `--lr` | `10` / `1e-4` | optimisation (Adam, constant learning rate) |
| `--image_size` | `64` | input resolution |
| `--channel` | `128 256 384 512` | channels of the U-Net levels (resolution halved between levels) |
| `--attention_resolutions` / `--num_heads` | `16` / `1` | resolutions with self-attention, number of heads |
| `--resume` | `False` | resume from `last.pt` |
| `--amp` | `False` | mixed precision |

The paper studies the influence of lambda (Fig. 9), of the threshold gamma (`--threshold_factor` of
`tools/evaluate.py`, Fig. 10), of the number of samples N (`--num_sample` / `--n_mask`, Fig. 11) and of the number of
DDIM steps (`--sample_step_ddim` of `tools/generate.py`, Table VIII).

## Citation

If you use this code, please cite:

```bibtex
@article{stenger2026unsupervised,
  title={Unsupervised domain adaptation in biomedical images segmentation with guided diffusion generative prior},
  author={Stenger, Alexandre and Baudrier, {\'E}tienne and Passat, Nicolas and Naegel, Beno{\^\i}t},
  journal={IEEE Transactions on Image Processing},
  volume={35},
  pages={3690--3704},
  year={2026},
  publisher={IEEE}
}
```

## Acknowledgements and license

This code is built on [IDDM: Integrated Design Diffusion Model](https://github.com/chairc/Integrated-Design-Diffusion-Model)
by chairc, and uses [Segment Anything](https://github.com/facebookresearch/segment-anything). Like IDDM, it is
released under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for details.
