"""
Predict flower name from an image with predict.py along with the probability of that name.

Usage:
    python predict.py /path/to/image checkpoint

Options:
    --top_k K                Return the top K most likely classes.
                             Example: python predict.py /path/to/image checkpoint --top_k 3

    --category_names FILE    Use a mapping of categories to real names.
                             Example: python predict.py /path/to/image checkpoint --category_names cat_to_name.json

    --gpu                    Use GPU for inference.
                             Example: python predict.py /path/to/image checkpoint --gpu

Arguments:
    /path/to/image           Path to the image file.
    checkpoint               Path to the model checkpoint file.

Description:
    This script predicts the flower name from an image along with the probability of that name.
    You need to provide the path to the image and the model checkpoint file.
    Additional options allow you to specify the number of top classes to return, use a category-to-name mapping file,
    and enable GPU for inference.
"""

# Required libraries
import argparse
import json
import torch
from torchvision import models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Predict flower name from an image with predict.py along with the probability of that name.")

    # Add arguments
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint file')
    parser.add_argument('--top_k', type=int, default=3, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Use a mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    # Parse arguments
    args = parser.parse_args()
    
    # Use the parsed arguments
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu

    # Print the parsed arguments (for debugging)
    print(f"Image Path: {image_path}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Top K: {top_k}")
    print(f"Category Names: {category_names}")
    print(f"Use GPU: {gpu}")

    #############################################################
    # The prediction code starts here
    #############################################################

    # Loads a checkpoint and rebuild the model
    def load_checkpoint(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        arch = checkpoint['architecture']
        model = models.get_model(arch, weights=None) # pretrained=False was deprecated
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        return model

    # PIL image preprocessing function for use in a PyTorch model
    def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        with Image.open(image) as im:
            # Resize
            im.thumbnail((256, 256))

            # Crop
            width, height = im.size
            left = (width - 224) / 2
            top = (height - 224) / 2
            right = left + 224
            bottom = top + 224
            im = im.crop((left, top, right, bottom))

            # Convert color channels to 0-1
            np_image = np.array(im) / 255

            # Normalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            im = (np_image - mean) / std

            # Reorder dimensions
            im = im.transpose((2, 0, 1))
        return im
    
    # Displays an image along with the top 5 classes
    def imshow(image, ax=None, title=None):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()
        
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.numpy().transpose((1, 2, 0))
        
        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        return ax
    
    # Translates an array of indices to their respective labels
    def translate_indices_to_labels(indices, class_to_idx):
        # Reverse mapping dictionary from index to class
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
        # Return the class names for the given indices
        return [idx_to_class[idx] for idx in indices]

    # Predicts the class from an image file
    def predict(image_path, model, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        device=torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        model.eval()
        model.to(device)
        image = process_image(image_path)
        image = torch.tensor(image).float()
        image = image.unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            output = model.forward(image)
            ps = torch.exp(output)
            top_p, top_class = ps.topk(topk, dim=1)
            indices = list(top_class.cpu().numpy().squeeze())
            top_class = translate_indices_to_labels(indices, model.class_to_idx)
        return top_p, top_class
    
    # Load the model from a checkpoint
    model = load_checkpoint(checkpoint)

    # Load the category names
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Display the image along with the top 5 classes
    image = process_image(image_path)
    top_p, top_class = predict(image_path, model)
    top_p = top_p.cpu().numpy().squeeze()
    # top_class = top_class.cpu().numpy().squeeze()
    classes = [cat_to_name[str(x)] for x in top_class]
    print(top_p)
    print(top_class)
    print(classes)

    # Display the top 5 classes, highest probability first
    fig, ax = plt.subplots()
    ax.barh(classes, top_p)
    ax.set_xlabel('Probability')
    ax.set_title('Top 5 classes')
    ax.invert_yaxis()
    plt.show()

    # Display the image
    print(image_path)
    imshow(torch.tensor(image))
    plt.title(cat_to_name[image_path.split('/')[-2]])
    plt.show()


if __name__ == "__main__":
    main()