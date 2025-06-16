# Tomato Leaf Disease Classification using _Convolutional Neural Network_ (CNN)

## Introduction
This is the repository for Research Based Learning (RBL) project from **SK5004 Pembelajaran Mesin dan Kecerdasan Buatan** lecture. This project is created by Melvan Savero Lee (10121063) and Habiburrohman (10121089) with the main objective to classify tomato leaf disease using deep learning method called Convolutional Neural Network (CNN). We hope that the CNN model produced from this project can help farmers to automate the tomato cultivation monitoring activities so the diseased plant can be detected and treated as soon as possible. For more information about this project, see [project diagram](./mermaid.md).

## Project Files
Below is the description of project files and folders.
1. Utility Files
- `dataset_builder.py`: Contain class definition to prepare the raw dataset from certain directory and convert it to TFRecord for the use with `tensorflow_datasets` (TFDS) library. 
- `utils.py`: Contain helper functions for preprocessing the dataset and building the CNN model. 
- `plots.py`: Contain specific functions for visualizing the results.
2. Notebook Files
- `binary_classification.ipynb`: Doing 2 class classification based on infection status (healhy, diseased)
- `quinary_classification.ipynb`: Doing 5 class classification based on pathogen type (health, bacteria, fungi, mite, virus)
- `main_classification.ipynb`: Doing 10 class classification based on pathogen species or equivalent to disease type (healthy, bacterial spot, early blight, late blight, leaf mold, target spot, septoria leaf spot, spider mite, tomato yellow leaf curl virus, tomato mosaic virus)
- `visualization.ipynb`: Visualize the training history and results from all models.
3. Folders
- `models`: Storing previously saved fitted model in `.keras` format.
- `logs`: Storing the data composition and training logs of the model in `.csv` format.
- `images`: Storing visualization of the results in `.png` format.

> Note that the notebook files have to import many class and functions from utility files.

## System Requirements (Recommendation)
- Windows 11 64 bit
- 12 GB RAM
- Python 3.10 
- Linux Terminal (for example, Git Bash)

## Guide
### Step 1: Repo Cloning and Dependency Installation
- Clone this repository and go to the repo directory.
- Create a python virtual environment named `pyenv`. <br>
```
python -m venv pyenv
```
- Enter the virtual environment. <br>
```
source pyenv/Scripts/activate
```
- Install the dependencies. <br>
```
pip install -r requirements.txt
```

### Step 2: Dataset Placing
- Go to the repo directory. 
- Create a folder named `dataset`. 
- Download the datasets. <br>
[tomato_leaf_disease_binary](https://www.kaggle.com/datasets/habiburrohman/tomato-leaf-disease-binary) <br>
[tomato_leaf_disease_quinary](https://www.kaggle.com/datasets/habiburrohman/tomato-leaf-disease-quinary) <br>
[tomato_leaf_disease_main](https://www.kaggle.com/datasets/habiburrohman/tomato-leaf-disease) <br>
- Extract each of the zipped dataset in the `dataset` folder, then rename them acccording to the hyperlink texts above.

### Step 3: Code Running
- Choose and open the classification notebooks, for example, `main_classification.ipynb` file.
- Run the code block by block. 
- If prefer not to train the model, skip the `Model Training` section and continue to `Model Loading` section for loading the existing fitted model.

> `Data Preparation` section produce the TFRecord file of compiled dataset and save it into `tfds` folder.

> `Data Splitting` section produce the class distribution of the dataset and save it into `logs` folder. This section consume large RAM in the background.

> `Model Training` section produce the training history of the model and save it into `models` folder.