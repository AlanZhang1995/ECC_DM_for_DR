# ECC_DM_for_DR

Official codebase for the MICCAI 2025 paper:
***Class-Conditioned Image Synthesis with Diffusion for Imbalanced Diabetic Retinopathy Grading***

---
Author Feedback according to Reviewer's question:

1) Implement choices. As our work focus on Diffusion Model development, the choices of datasets and classifiers follow fair and effecient DM evaluation purpose. We selected DDR as the primary dataset due to its defined public test split and availability of strong open-source baselines. EyePACS lacks a public test label. Even if used as the main dataset, EyePACS would only reduce the gain from DreamBooth, but our core innovation—semantic filtering and ECC—should remain beneficial. For classifiers, we prioritized established methods rather than custom networks to better isolate the impact of our DM. LANet[10] itself offers SOTA Acc. with diverse 5 backbones, making it ideal for integration. Our method improves performance across top 3 backbones, demonstrating its backbone-independence. [R2Q2] To reduce potential bias from any single classifier, we use a diverse ensemble and it works well in our experiment. In more challenging case where classifier quality is limited, we will consider human-in-the-loop filtering a promising direction.

2) Our model can be trained and inference on a single RTX 4090 (24GB)

---

## Environment

We use the official PyTorch Docker image:

```bash
docker pull pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
```

An example for creating a container is provided in `start_docker.sh` (modify the `-v path` according to your setup).

Alternatively, you may pull our pre-built Docker image (note: very large; recommend using above example):

```bash
docker pull haochen1995/ecc-dm-for-dr
```

---

## Data Structure

A screenshot of the expected data organization is provided under the `data_structure/` folder, along with the necessary CSV files.

* Datasets should be downloaded from their official websites.
* `DR_grading/` contains my train/validation/test split (to be merged with your DDR dataset).
* Pretrained **LANet checkpoints** (used for semantic quality evaluation) are available here: [Google Drive](https://drive.google.com/file/d/1Wkii7c3O-amhQJjubHUMsUn4CkjMFbQQ/view?usp=sharing).

> **Note:** We preprocess images following the LANet paper using [this method](https://github.com/ErikLarssonDev/PyTorch/blob/011d4de2dbea1a09cbaa1608ce04b7411f6730f3/kaggle_diabetic_retinopathy_detection/preprocess_images.py#L57) or our example `./data_structure/DR_grading/preprocess_images.py`. Preprocessing is optional but helps reproduce LANet results.

---

## Testing

### LANet Evaluation (Accuracy reported in paper)

Download our trained LANet models (trained with DM data, ECC w/o filter, VGG backbone, 5-fold):
[Google Drive](https://drive.google.com/file/d/1a-el5k1JCOWMMLV7omp-VrEUlSpVXhd2/view?usp=sharing)

```bash
cd LANet
mkdir logs
# Update model path at Line 210
python main_lanet.py --model vgg --adaloss True --visname tests --test True
```

---

## End-to-End Pipeline

We verified the entire pipeline works as intended. You may encounter minor path issues that require adjustment.

### 1. Diffusion Model Training

We train separate models for each DR stage.

Example training script: `./Diffusion_Model/run_fundus.sh`
Modify `--stage 4` to select a specific stage.

```python
label2stage = {
    0: 'normal',
    1: 'mild DR',
    2: 'moderate DR',
    3: 'severe DR',
    4: 'proliferative DR'
}
```

```bash
# Modify dataset path (around Line 58 of Retina_datasets.py)
cd Diffusion_Model
bash run_fundus.sh
```

**Additional hyperparameters:**

```python
parser.add_argument('--cvs_file', type=str, default='.../train_3fundus.csv')  # CSV file under data_structure/
parser.add_argument('--stage', type=int, default=None, help='Train with specific DR stage')
parser.add_argument('--prior_loss_weight', type=float, default=1)
parser.add_argument('--sync_loss_weight', type=float, default=1)
parser.add_argument('--sync_loss_start_step', type=int, default=5000)
parser.add_argument('--sync_loss_interval', type=int, default=100)
parser.add_argument('--sync_outdir', type=str, default='./Sync_dataset_fundus/')
```

### 2. Sync Data Generation (Diffusion Model Inference)

```bash
# Update settings in Line 12–20
# Update model path at Line 40
cd Diffusion_Model
python Fundus_test.py
```

### 3. LANet with Sync Data

Filter and randomly sample generated data.
Example script: `./LANet/select_topK_sync.py`

### 4. LANet Training with Sync Data

```bash
cd LANet
# Update dataset path at Line 19 of ddrdataset.py
bash train.sh
```

**Additional hyperparameters**

```python
parser.add_argument('--fold', type=int, default=None) # 5-fold validation experiments, 0-4
parser.add_argument('--use_sampler', action='store_true', help='use weighted sampler to balance data')
parser.add_argument('--use_sampler_weight', type=int, default=0, help='adjust the sample weight for oversampling')
parser.add_argument('--sync_data', type=str, default=None, help='w/ and w/o filtering')
parser.add_argument('--ratio', type=float, default=None, help='useless in this version')
```

---

## Citation

```bibtex
@inproceedings{Placeholder,
  ...
}
```