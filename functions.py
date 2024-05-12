import numpy as np
import matplotlib.pyplot as plt
import ML_DL_Functions3
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import glob
import os
import itertools

def create_np_data_array(num_of_triplets, data_set, file_names):
    """
    Creates a numpy array from a dataset for the specified number of triplets.

    Parameters:
    - num_of_triplets: Number of triplets to consider.
    - data_set: Dictionary containing image data for all triplets.
    - file_names: List of file names corresponding to the images in the dataset.

    Returns:
    - new_data: Numpy array of shape [num_of_triplets, 3, 2, 224, 224, 3] containing the image data.
    """
    new_data = np.zeros((num_of_triplets, 3, 2, 224, 224, 3))

    for i in range(num_of_triplets):
        for j in range(3):
            for k in range(2):
                new_data[i, j, k] = (data_set[file_names[6*i+2*j+k]] / 255) - 0.5
    return new_data


def read_data_from_file(path):
    """
    Reads image data from files in the specified path.

    Parameters:
    - path: Path to the folder containing image files.

    Returns:
    - images: Dictionary containing image data.
    """

    images = {}

    for file in sorted(glob.glob(path)):
        filename = file.split("/")[-1]            # get the name of the .jpg file
        img = np.asarray(Image.open(file))        # read the image as a numpy array
        images[filename] = img[:, :, :3]          # remove the alpha channel

    return images


def generate_same_pair(data_set):
    """
    Concatenates left and right shoe images along the height axis for each pair in the data set.

    Parameters:
    - data_set: numpy array of shape [*, 3, 2, 224, 224, 3], representing a dataset of triplets.

    Returns:
    - concatenated_pairs: numpy array of shape [*, 448, 224, 3], where each pair of shoes is concatenated along the height axis.
    """
    # Get the shape of the input data_set
    num_triplets, num_pairs, num_shoes, height, width, channels = data_set.shape

    # Initialize the output array
    concatenated_pairs = np.zeros((num_triplets * num_pairs, 448, 224, 3))

    # Iterate over triplets, pairs, and shoes to concatenate left and right shoe images
    for i in range(num_triplets):
        for j in range(num_pairs):
            left_shoe = data_set[i, j, 0]
            right_shoe = data_set[i, j, 1]

            # Concatenate left and right shoe images along the height axis
            concatenated_pair = np.concatenate((left_shoe, right_shoe), axis=0)

            # Store the concatenated pair in the output array
            concatenated_pairs[i * num_pairs + j] = concatenated_pair

    return concatenated_pairs


def generate_different_pair(data_set):
    """
    Generates a numpy array where each image contains 2 shoes from different pairs but submitted by the same student.

    Parameters:
    - data_set: numpy array of shape [*, 3, 2, 224, 224, 3], representing a dataset of triplets.

    Returns:
    - different_pairs: numpy array of shape [*, 448, 224, 3], where each image contains 2 shoes from different pairs but submitted by the same student.
    """
    # Get the shape of the input data_set
    num_triplets, num_pairs, num_shoes, height, width, channels = data_set.shape

    # Initialize the output array
    different_pairs = np.zeros((num_triplets * num_pairs, 448, 224, 3))

    # Define the two possible permutations
    permutations = [np.array([2, 0, 1]), np.array([1, 2, 0])]

    # Iterate over triplets to generate different pairs
    for i in range(num_triplets):
        # Choose randomly between the two permutations
        jumbled_indices = np.random.choice(len(permutations))

        for j in range(num_pairs):
            permutation = permutations[jumbled_indices]
            left_shoe = data_set[i, permutation[j], 0]
            right_shoe = data_set[i, j, 1]

            # Concatenate left and right shoe images along the height axis
            different_pair = np.concatenate((left_shoe, right_shoe), axis=0)

            # Store the different pair in the output array
            different_pairs[i * num_pairs + j] = different_pair

    return different_pairs


def get_accuracy(model, data, batch_size=50,device='cpu'):
    """Compute the model accuracy on the data set. This function returns two
    separate values: the model accuracy on the positive samples,
    and the model accuracy on the negative samples.

    Example Usage:

    >>> model = CNN() # create untrained model
    >>> pos_acc, neg_acc= get_accuracy(model, valid_data)
    >>> false_positive = 1 - pos_acc
    >>> false_negative = 1 - neg_acc
    """

    model.eval()
    n = data.shape[0]

    data_pos = generate_same_pair(data)      # should have shape [n * 3, 448, 224, 3]
    data_neg = generate_different_pair(data) # should have shape [n * 3, 448, 224, 3]

    pos_correct = 0
    for i in range(0, len(data_pos), batch_size):
        xs = torch.Tensor(data_pos[i:i+batch_size]).permute(0,3,1,2)
        xs = xs.to(device)
        zs = model(xs)
        pred = zs.max(1, keepdim=True)[1] # get the index of the max logit
        pred = pred.detach().cpu().numpy()
        pos_correct += (pred == 1).sum()

    neg_correct = 0
    for i in range(0, len(data_neg), batch_size):
        xs = torch.Tensor(data_neg[i:i+batch_size]).permute(0,3,1,2)
        xs = xs.to(device)
        zs = model(xs)
        pred = zs.max(1, keepdim=True)[1] # get the index of the max logit
        pred = pred.detach().cpu().numpy()
        neg_correct += (pred == 0).sum()

    return pos_correct / (n * 3), neg_correct / (n * 3)


def train_model(model,
                train_data,
                validation_data,
                batch_size=32,
                learning_rate=0.001,
                weight_decay=1e-4,
                epochs=20,
                checkpoint_path=None):
    """
    Train a CNN model for the shoe matching task.

    Parameters:
    - model: The Siamese CNN model to be trained.
    - train_data: The training dataset (positive and negative pairs).
    - validation_data: The validation dataset for monitoring performance.
    - batch_size: The number of samples in each mini-batch.
    - learning_rate: The learning rate for the optimizer.
    - weight_decay: The L2 regularization strength.
    - epochs: The number of training epochs.
    - checkpoint_path: The path to save the model checkpoints.

    Returns:
    - training_curve: A dictionary containing training and validation accuracies and losses over epochs.
    """

    # Initialize Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Create the positive and negative pairs
    positive_pairs = generate_same_pair(train_data)
    negative_pairs = generate_different_pair(train_data)
    
    # Containers to store losses and accuracies
    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []
    train_loss_epoch = []
    valid_loss_epoch_mean = []
    update_lr_flag = 1

    for epoch in range(epochs):
        # Shuffle the positive and negative pairs
        np.random.shuffle(positive_pairs)
        np.random.shuffle(negative_pairs)

        # Reset the accuracy containers at the beginning of each epoch
        epoch_train_accuracies = []
        epoch_validation_accuracies = []

        for i in range(0, len(positive_pairs) - batch_size // 2, batch_size // 2):
            # Sample batch_size//2 of positive pairs and batch_size//2 of negative pairs
            positive_batch = positive_pairs[i:i + batch_size // 2]
            negative_batch = negative_pairs[i:i + batch_size // 2]

            # Create the labels of the batch and combine the positive and negative half batches
            labels = np.concatenate([np.ones(batch_size // 2), np.zeros(batch_size // 2)])
            
            # Concatenate positive and negative pairs for input
            batch_input = np.concatenate([positive_batch, negative_batch])

            # Get the number of samples
            num_samples = len(labels)

            # Generate a random permutation of indices
            perm = np.random.permutation(num_samples)

            # Shuffle batch_input and labels using the same permutation
            shuffled_batch_input = batch_input[perm]
            shuffled_labels = labels[perm]
            
            # Conversion from numpy array to torch tensor (NCHW format)
            input_tensor = torch.from_numpy(shuffled_batch_input).permute(0, 3, 1, 2).float()
            labels_tensor = torch.from_numpy(shuffled_labels).long()

            # Reset the optimizer
            optimizer.zero_grad()
            # Predict output
            output = model(input_tensor)
            # Compute the loss
            loss = criterion(output, labels_tensor)
            # Backward pass
            loss.backward()
            # Update the parameters
            optimizer.step()
            # Append loss and accuracy for tracking
            train_losses.append(loss.item())
            accuracy = ((output.argmax(dim=1) == labels_tensor).sum().item()) / labels_tensor.size(0)
            epoch_train_accuracies.append(accuracy)

        # Calculate validation accuracy using get_accuracy function
        validation_pos_acc, validation_neg_acc = get_accuracy(model, validation_data, batch_size=batch_size)
        validation_mean_acc = (validation_pos_acc + validation_neg_acc) / 2

        # Update lists for tracking
        epoch_validation_accuracies.append(validation_mean_acc)
        validation_accuracies.append(validation_mean_acc)
        train_accuracies.append(np.mean(epoch_train_accuracies))
        train_loss_epoch.append(np.mean(train_losses))

        if update_lr_flag and (((validation_mean_acc > 0.88) and (model._get_name() == 'CNNChannel')) or ((validation_mean_acc > 0.84) and (model._get_name() == 'CNN')) or np.mean(epoch_train_accuracies) > 0.91):
            update_lr_flag = 0
            new_learning_rate = learning_rate / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_learning_rate
            print(f"Learning rate decayed to {new_learning_rate}")

        
        # Checkpoint the model
        if checkpoint_path:
            torch.save(model.state_dict(), f"{checkpoint_path}/model_epoch_{epoch}.pk")

        # Display epoch statistics
        print(f"Epoch {epoch + 1}/{epochs} => "
            f"Train Loss: {np.mean(train_losses):.4f}, "
            f"Train Accuracy: {np.mean(epoch_train_accuracies) * 100:.2f}%, "
            f"Validation Accuracy: {validation_mean_acc * 100:.2f}%")
        
        # Calculate validation loss and accuracy
        validation_input = np.concatenate([generate_same_pair(validation_data),
                                        generate_different_pair(validation_data)])
        validation_labels = np.concatenate([np.ones(len(validation_data) * 3),
                                            np.zeros(len(validation_data) * 3)])
        validation_input_tensor = torch.from_numpy(validation_input).permute(0, 3, 1, 2).float()
        validation_labels_tensor = torch.from_numpy(validation_labels).long()
        validation_output = model(validation_input_tensor)
        validation_loss = criterion(validation_output, validation_labels_tensor)
        validation_losses.append(validation_loss.item())
        valid_loss_epoch_mean.append(np.mean(validation_losses))

    # Return training curve information
    training_curve = {
        'train_losses': train_loss_epoch,
        'train_accuracies': train_accuracies,
        'validation_losses': valid_loss_epoch_mean,
        'validation_accuracies': validation_accuracies
    }
    return training_curve


# Function to plot training curves
def plot_training_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot training and validation losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(valid_losses, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()

    # Plot training and validation accuracies
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(valid_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()
