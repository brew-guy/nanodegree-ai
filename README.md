# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Repository Structure

The repository contains the following files and directories:

- `assets/`: Contains additional resources such as `cat_to_name.json` which maps category labels to names.
- `flower_data/`: Directory for storing flower images used for training and validation.
- `Image Classifier Project Take2.html`: HTML export of the Jupyter Notebook used for the project.
- `Image Classifier Project Take2.ipynb`: Jupyter Notebook containing the project code and explanations.
- `LICENSE`: License file for the project.
- `predict.py`: Script for predicting the class of an input image using the trained model.
- `README.md`: This file, providing an overview of the project and repository structure.
- `train.py`: Script for training the image classifier model.
- `*.pth`: Checkpoint files for the trained models.

### Image Folder Structure

The image dataset should be organized into a directory structure that separates the images into training, validation, and testing sets. Each set should have subdirectories for each class of images. The structure should look like this:

```
image_folder/
    train/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            image2.jpg
            ...
        ...
    valid/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            image2.jpg
            ...
        ...
    test/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            image2.jpg
            ...
        ...
```

### Prerequisites

- Python 3.x
- PyTorch
- torchvision
- tqdm

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/brew-guy/nanodegree-ai
   cd aipnd-project
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

### Training the Model

To train the model, run the `train.py` script with the appropriate arguments. For example:

```sh
python train.py flower_data --arch "vgg16" --learning_rate 0.001 --hidden_units 512 --epochs 10 --gpu
```

### Predicting Image Classes

To predict the class of an image using the trained model, run the predict.py script with the appropriate arguments. For example:

```sh
python predict.py path/to/image checkpoint.pth --top_k 5 --category_names "cat_to_name.json" --gpu
```

### Project Steps

The project is broken down into multiple steps:

1. Load and preprocess the image dataset.
2. Train the image classifier on the dataset.
3. Use the trained classifier to predict image content.

### Acknowledgements

This project is part of Udacity's AI Programming with Python Nanodegree program.
