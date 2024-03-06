# Deep Learning-Based Pneumothorax Diagnosis and Segmentation System for Portable Supine Chest X-rays

## About The Project
This is the original implementation of "Deep Learning-Based Pneumothorax Diagnosis and Segmentation System for Portable Supine Chest X-rays".



### Built with
* [MONAI](https://monai.io/)
* [PyTorch](https://pytorch.org/)
* [Segmentation Models](https://segmentation-models.readthedocs.io/en/latest/)

### Requirements
* [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
* GPU with at least 15GB VRAM

## Installation
1. Clone the repo
   ```sh
   $ git clone https://github.com/r09246006/NTUH-Pneumothorax-Segmentation.git
   ```
2. Create new conda environment
   ```sh
   $ conda create -n oxr python==3.8
   $ conda activate oxr
   ```
3. Install required packages
   ```sh
   $ pip3 install -r requirements.txt
   ```

## Tutorial
The diagnostic deep learning system use the two stage system, a classification model follow by a segmentation model. For performance benchmark, we bulid a CNN- and a Transformer-based system for comparison ("EfficientNetV2" and "TinyViT").
The development of diagnostic deep learning system consist of three phases: pre-training, fine-tuning and testing.


### Step.1 Pre-training phase
All classification and segmentation models were first pre-trained on SIIM-ACR (Society for Imaging Informatics in Medicine, American College of Radiology) Pneumothorax Segmentation dataset.

The pre-training process will obtained pre-trained weights of the segmentation model for fine-tuning of the segmentation models, and pre-trained weights of the classification model for fine-tuning of the classification models. Additionally, the pre-trained segmentation model weights were also utilized during
the pre-training phase of the classification models for weight initialization. This design aimed to increase training speed and provide a smoother sampling rate transition, further details please refer my paper, section2.4.1. 

1. Start training segmentation model:\
    The model will (automatically) initialize from ImageNet pre-trained model weight.
    ```sh
    cd ./step1_pretraining/step1_segmentation
    bash train.sh
    ```
    you can specify which GPU to use
    ```sh
    bash train.sh 0
    ```
    use nvtop to see which GPU is currently available
    ```sh
    nvtop
    ```


2. Start training classification model:\
    The model will (automatically) initialize from above trained segmentation model checkpoint, the training will not proceed if the model checkpoint does not exist, make sure the above segmentation model training is finished.

   ```sh
    cd ./step1_pretraining/step2_classification
    bash train.sh
    ```

3. After finish training, results will be available in `runs/`

### Step.2 NTUH fine-tuning phase

After pre-training, both the classification and segmentation models were fine-tuned on the NTUH dataset. This phase will obtained classification model weights, segmentation model weights, classification threshold, segmentation threshold. The weights of the classification and segmentation models were initialized with the respective pre-trained model weights obtained from the pre-training phase. The training order of classification and segmentation will not affect the outcome.

1. Start training segmentation model:\
    The model will (automatically) initialize from the trained segmentation model checkpoint (pre-training phase), the training will not proceed if the model checkpoint does not exist, make sure the pre-training phase is finished.
    ```sh
    cd ./step2_ntuh_training/ntuh_segmentation
    bash train.sh
    ```


2. Start training classification model:\
    The model will (automatically) initialize from the trained classification model checkpoint (pre-training phase), the training will not proceed if the model checkpoint does not exist, make sure the pre-training phase is finished.

   ```sh
    cd ./step2_ntuh_training/ntuh_classification
    bash train.sh
    ```

3. After finish training, results will be available in `runs/`

4. Start calculating the optimal thresholds for classification and segmentation models:
    ```sh
    cd ./step2_ntuh_training/get_thres
    bash get_thresholds.sh
    ```
    Thresholds of classification and segmentation will store in `cls_vit_5fold_result/5cv_avg.json` and `cls_eff_5fold_result/5cv_avg.json`


### Step.3 NTUH testing phase

The inference process of the diagnostic deep learning system utilizes classification model checkpoints, segmentation model checkpoints, classification threshold, segmentation threshold obtained from the fine-tuning phase.

1. Start testing:\
    All models and thresholds will (automatically) initialize from results of the fine-tuning phase. \
    model checkpoints: `./step2_ntuh_training/ntuh_segmentation/model_weights` , `./step2_ntuh_training/ntuh_classification/model_weights`\
    model thresholds: `./step2_ntuh_training/get_thres/cls_vit_5fold_result/5cv_avg.json` , `./step2_ntuh_training/get_thres/cls_eff_5fold_result/5cv_avg.json`\
    The testing will not proceed if the fine-tuning phase is not finished and generate the required checkpoints and thresholds.


    ```sh
    cd ./step3_testing/testing
    bash inference.sh
    ```

    The classification & segemntation testing result temp file will stored in `./step3_testing/testing`, this can avoid repeat GPU usage if recomputing the testing result is required.

2. Start computing performance metric of        classification and segmentation:

    The python script will read the testing result temp file from above step

    ```sh
    cd ./step3_testing/testing
    python bootstrap.py
    ```
    All evaluation metric, point estimates, 95% condident intervals, will show on terminal screen.
