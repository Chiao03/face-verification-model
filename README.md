## Project description
This project investigates the effect of different feature extraction techniques on the robustness of face verification algorithms, specifically Support Vector Machine (SVM), SVM combined with Convolutional Neural Networks (SVM-CNN), and Hybrid Vision Transformers (Hybrid-ViT). The robustness is evaluated by comparing performance on aligned and distorted face datasets derived from the Labeled Faces in the Wild (LFW) dataset.

## Dataset
The Labeled Faces in the Wild (LFW) dataset is used, which contains more than 13,000 images of faces under various conditions (e.g., lighting, poses, backgrounds). The dataset is used to create two different datasets for the experiments:

  - **Aligned Dataset**: Contains aligned and cropped images.
  - **Distorted Dataset**: Aligned images with added Gaussian noise or occlusions to simulate real-world challenges.

## Prerequisites

- Operating System: Windows, macOS, or Linux
- Python Version: 3.11 or higher
- Conda: Ensure that Conda is installed on your system. If not, download and install it from [Anaconda](https://docs.anaconda.com/anaconda/install/).
- Jupyter Notebook (for visualization): Included with Anaconda or install separately.


## How to run

### 1. Setup environment
```
conda env create -f environment.yml
conda activate envsml
```
### 2. Preprocessing

#### Download and config dataset

- Download the LFW dataset from [LFW Dataset Link](https://vis-www.cs.umass.edu/lfw/).
- Extract the LFW dataset and rename the extracted folder to `lfw`. Place this `lfw` folder at the root of the project.

#### Create aligned face dataset
- Run the following commands for create the aligned face dataset from lfw dataset
    ```
    cd preprocessing
    python alignfaces.py

    ```
#### Create distorted face dataset
- Run the following commands for create the distored face dataset from aligned face dataset
    ```
    cd preprocessing
    python distortedface.py

    ```
### 3. Model training

#### 3.1. SVM

- If you want to use the default folders and file paths of our project, just run the following command:

    ```
    python svm.py 

    ```

- If you want to customize the input parameters (e.g., using a different data folder or result file name), please run the command below with your desired paths:
    ```
    python svm.py --pairfile pairs.txt --datafolder data/alignedfaces --resultfolder results --resultfile svm_aligned

    ```

#### 3.2. SVM-CNN
- If you want to use the default folders and file paths of our project, just run the following command:
    ```
    python svm_cnn.py 

    ```
- If you want to customize the input parameters (e.g., using a different data folder or result file name), please run the command below with your desired paths:
    ```
    python svm_cnn.py --pairfile pairs.txt --datafolder data/alignedfaces --resultfolder results --resultfile cnn_aligned

    ```

#### 3.3. Hybrid ViT

- Please download the model from [this link](https://drive.google.com/file/d/1vXV3NT35loFR984zwahi7LSlpDM3-U00/view?usp=sharing) before training the HybridViT and place it in the current directory (./).  
The pre-trained model is from Phan, H., Le, C., Le, V., He, Y., & Nguyen, A. (2024). *Fast and Interpretable Face Identification for Out-Of-Distribution Data Using Vision Transformers*. https://doi.org/10.1109/WACV57701.2024.00618

- If you want to use the default folders and file paths of our project, just run the following command:
    ```
    python hybrid_vit_cross_validation.py 

    ```
- If you want to customize the input parameters (e.g., using a different data folder or result file name), please run the command below with your desired paths:
    ```
    python hybrid_vit_cross_validation.py --pairfile pairs.txt --datafolder data/alignedfaces --resultfolder results --resultfile hybridvit_aligned

    ```

### 4. Result visualization
- Please run the below command to start Jupyter Notebook:
    ```
    jupyter notebook

    ```
  The above command will open a window where you can navigate to display_result.ipynb to see the visualisation of the results

## File description

```
├── data                                # This folder is not available in submission
|                                         please create it.
|                                         Folder containing raw and processed face image data
│   ├── alignedfaces                    # Directory containing aligned and cropped face images
│   └── distortedfaces                  # Directory containing face images with distortions 
│
├── preprocessing                       # Folder containing preprocessing scripts for the LFW dataset
│   ├── alignfaces.py                   # Script to detect, align, and crop faces 
|                                         from raw images using MTCNN
│   ├── data_creator.py                 # Script to create pairs of images 
|                                         (matched and mismatched) for face verification
│   └── distortfaces.py                 # Script to apply distortions
│
├── results                             # Folder storing results of all models
│   ├── cnn_aligned_labels.csv          # Label file for the aligned dataset used by the CNN model
│   ├── cnn_aligned.csv                 # CSV file containing features for the aligned dataset (CNN)
│   ├── cnn_distorted_labels.csv        # Label file for the distorted dataset used by the CNN model
│   ├── cnn_distorted.csv               # CSV file containing features for the distorted dataset (CNN)
│   ├── svm_aligned_labels.csv          # Label file for the aligned dataset used by the SVM model
│   ├── svm_aligned.csv                 # CSV file containing features for the aligned dataset (SVM)
│   ├── svm_distorted_labels.csv        # Label file for the distorted dataset used by the SVM model
│   ├── svm_distorted.csv               # CSV file containing features for the distorted dataset (SVM)
│   ├── hybridvit_aligned_labels.csv    # Label file for the aligned dataset used by the Hybrid ViT model
│   ├── hybridvit_aligned.csv           # CSV file containing features for the aligned dataset (Hybrid ViT)
│   ├── hybridvit_distorted_labels.csv  # Label file for the distorted dataset used by the the Hybrid ViT model
│   ├── hybridvit_distorted.csv         # CSV file containing features for the distorted dataset (Hybrid ViT)
│   └── *.log                           # Log files storing detailed output and progress of each run
│
├── .gitignore                          # Specifies which files and folders should be ignored by Git version control
│
├── metric_calculator.py                # Script to calculate various evaluation metrics 
|                                        (e.g., accuracy, precision, recall)
│
├── nested_cross_validation.py          # Main script to implement 20-fold nested cross-validation
│
├── pairs.txt                           # Text file containing pairs of images and their labels (matched/mismatched)
│
├── README.md                           # Project documentation file with an overview, setup instructions, and usage guidelines
│
├── svm_cnn.py                          # Implementation of the SVM-CNN hybrid model using ResNet-50 as the feature extractor
│
├── cnn_resnet50.py                     # Implementation of ResNet-50 model for CNN
│
├── svm.py                              # Implementation of the baseline SVM model using HOG (Histogram of Oriented Gradients) for feature extraction
│
├── hybrid_vit.py                       # Implementation of Hybrid ViT
│
├── hybrid_vit_cross_validation.py      # Implementation of Hybrid ViT training with cross validation
│
├── hybrid_vit_transform_data.py        # Implementation of transforming data directory to a tuple with image and label
│
├── resnet18.py                         # Implementation of ResNet18 Face Model
│
└── display_result.ipynb                # Display visualisation results for all models on original and distorted data
```

## Report
You can find the detailed report [here](./COMP90051-Group2-report.pdf).

## References

- [Dataset LFW](https://vis-www.cs.umass.edu/lfw/)
- [Face-vit repo](https://github.com/anguyen8/face-vit)
- [ResNet model](https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py)
