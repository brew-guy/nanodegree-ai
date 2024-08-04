"""
Train a new network on a data set with train.py

Basic usage:
    python train.py data_directory

Prints out training loss, validation loss, and validation accuracy as the network trains.

Options:
    * Set directory to save checkpoints:
        python train.py data_dir --save_dir save_directory
    * Choose architecture:
        python train.py data_dir --arch "vgg13"
    * Set hyperparameters:
        python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    * Use GPU for training:
        python train.py data_dir --gpu
"""

# Required libraries
import argparse
import json
import os
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from tqdm import tqdm

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Use train.py to train a new network on an image dataset.")

    # Add arguments
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (default: vgg16)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units (default: 4096)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    # Parse arguments
    args = parser.parse_args()

    # Use the parsed arguments
    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu

    # Print the parsed arguments (for debugging)
    print(f"Data Directory: {data_dir}")
    print(f"Save Directory: {save_dir}")
    print(f"Architecture: {arch}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Hidden Units: {hidden_units}")
    print(f"Epochs: {epochs}")
    print(f"Use GPU: {gpu}")

    #############################################################
    # The training code starts here
    #############################################################

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    # Setting transforms for the training and validation sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # Load datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Define the dataloaders with image datasets and transforms
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    # Load a mapping of categories to names (must exist in same directory as train.py)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    print(len(cat_to_name))

    # Load a pre-trained network
    # Show loading of the model with a progress bar
    with tqdm(total=1, desc=f"Loading model {arch}") as pbar:
        model = models.get_model(arch, weights="DEFAULT") # Later version no longer support pretrained=True
        pbar.update(1)

    # Print the name of the default pretrained weights that are being used
    # TODO: Figure out how to list the 
    weights_enum = models.get_model_weights(arch)
    weights_names = ", ".join([member.name for member in weights_enum])
    print(f"Using weights: {weights_enum.__name__, weights_names}")

    # Freeze parameters to avoid messing with them when backpropagating through the full architecture
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, hidden_units)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(p=0.5)),
                            #   ('fc2', nn.Linear(hidden_units, hidden_units)),
                            #   ('relu2', nn.ReLU()),
                            #   ('dropout2', nn.Dropout(p=0.5)),
                            ('fc3', nn.Linear(hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
        
    model.classifier = classifier

    # Train using GPU if it's selected and available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device);

    # Print message to indicate that the model is being trained on the GPU/CPU
    print(f"Training started on {'GPU' if device.type == 'cuda' else 'CPU'}")

    # Train the classifier parameters while keeping feature parameters frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # epochs = 10
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                valid_accuracy = 0
                model.eval()
                with torch.no_grad():

                    # Validation
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        valid_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {valid_accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

    # Join save_dir with the model architecture and save the model as a checkpoint
    save_path = os.path.join(save_dir, f"{arch}_checkpoint.pth")

    checkpoint = {
        'architecture': arch,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': train_data.class_to_idx
    }

    torch.save(checkpoint, save_path)
    print(f"Model saved as {save_path}")


if __name__ == "__main__":
    main()