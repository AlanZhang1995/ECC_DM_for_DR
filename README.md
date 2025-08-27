# ECC_DM_for_DR
Offical codebase for MICCAI 2025 paper "_Class-Conditioned Image Synthesis with Diffusion for Imbalanced Diabetic Retinopathy Grading_"

---
Author Feedback according to Reviewer's question:

1) Implement choices. As our work focus on Diffusion Model development, the choices of datasets and classifiers follow fair and effecient DM evaluation purpose. We selected DDR as the primary dataset due to its defined public test split and availability of strong open-source baselines. EyePACS lacks a public test label. Even if used as the main dataset, EyePACS would only reduce the gain from DreamBooth, but our core innovation—semantic filtering and ECC—should remain beneficial. For classifiers, we prioritized established methods rather than custom networks to better isolate the impact of our DM. LANet[10] itself offers SOTA Acc. with diverse 5 backbones, making it ideal for integration. Our method improves performance across top 3 backbones, demonstrating its backbone-independence. [R2Q2] To reduce potential bias from any single classifier, we use a diverse ensemble and it works well in our experiment. In more challenging case where classifier quality is limited, we will consider human-in-the-loop filtering a promising direction.

2) Our model can be trained and inference on a single RTX 4090 (24GB)

---

## Environment
Basically, I am using the pytorch official docker image `pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime` with additional libs.

An example of creating a docker can be found in `start_docker.sh`, need to change `-v path` according to your need.

I also upload my own docker image just in case `docker pull haochen1995/ecc-dm-for-dr`. But it is super huge.

## Data Structure
I provide one screenshot of my data structure under `data_structure` folder along with necessary csv files.

`DR_grading` includes my train/valid/test split. Please merge it with your DDR dataset.

The pretrained LANet ckpts (the classifier group for semantic quality evaluation) can be found here: [google drive](https://drive.google.com/file/d/1Wkii7c3O-amhQJjubHUMsUn4CkjMFbQQ/view?usp=sharing).

Download the datasets from their official website. BTW, I preprocessed the image following the LANet paper with [method](https://github.com/ErikLarssonDev/PyTorch/blob/011d4de2dbea1a09cbaa1608ce04b7411f6730f3/kaggle_diabetic_retinopathy_detection/preprocess_images.py#L57) or my example code `./data_structure/DR_grading/preprocess_images.py`. I did so to better reproduce LANet performance. Please choose whether to do so based on your needs.


## Testing

### LANet Evaluation (Accuracy in paper)

Download our trained LANet models (with DM data, ECC w/o filter, vgg, 5-fold): [google drive](https://drive.google.com/file/d/1a-el5k1JCOWMMLV7omp-VrEUlSpVXhd2/view?usp=sharing)

```bash
cd LANet
mkdir logs
# change model path in Line 210
python main_lanet.py --model vgg --adaloss True --visname tests --test True
```

## Whole Pipline

I tested run the code through the whole pipline. It should good for you too run after fixing some _might unexpected_ path issue.

### 1. Diffusion Model Training

In practice, I trained seperate models for different DR stages.

An example of training script is `./Diffusion_Model/run_fundus.sh`, modify `--stage 4` to train DM with desered data.

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
# modify dataset path around Line 58 of Retina_datasets.py
cd Diffusion_Model
bash run_fundus.sh
```

PS: new hyperparameters
```python
parser.add_argument('--cvs_file', type=str, default='/home/haochen/DataComplex/universe/DataSet/diabet_fundus/train_3fundus.csv', help='load data split cvs file') #file under data_structure folder
parser.add_argument('--stage', type=int, default=None, help='Train the classifier with given DR stage')
parser.add_argument('--prior_loss_weight', type=float, default=1, help='loss weight for prior dataset')
parser.add_argument('--sync_loss_weight', type=float, default=1, help='loss weight for sync dataset')
parser.add_argument('--sync_loss_start_step', type=int, default=5000, help='iteration for generate sync samples')
parser.add_argument('--sync_loss_interval', type=int, default=100, help='iteration for update sync samples')
parser.add_argument('--sync_outdir', type=str, default='./Sync_dataset_fundus/', help='sync data dir for ECC')
```

### 2. Sync Data Generation (Diffusion Model Inference)
```bash
# change settings in Line 12-20
# change model path in Line 40
cd Diffusion_Model
python Fundus_test.py
```

### 3. LANet with Sync Data
filter and ramdom sample data from all generated data, example code is `./LANet/select_topK_sync.py`

### 4. LANet with Sync Data

```bash
cd LANet
# change dataset path in Line 19 of ddrdataset.py
bash train.sh
```

```python
parser.add_argument('--fold', type=int, default=None) # 5-fold validation experiments, 0-4
parser.add_argument('--use_sampler', action='store_true', help='use weighted sampler to balance data')
parser.add_argument('--use_sampler_weight', type=int, default=0, help='adjust the sample weight for oversampling')
parser.add_argument('--sync_data', type=str, default=None, help='w/ and w/o filtering')
parser.add_argument('--ratio', type=float, default=None, help='useless in this version')
```