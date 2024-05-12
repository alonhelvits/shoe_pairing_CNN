import matplotlib.pyplot as plt
import ML_DL_Functions3
import torch
import os
import itertools
import functions as fn

# Get the current working directory
current_directory = os.getcwd()

train_path =  os.path.join(current_directory, "data/train/*.jpg")
train_images = fn.read_data_from_file(train_path)

test_m_path =  os.path.join(current_directory, "data/test_m/*.jpg")
test_m_images = fn.read_data_from_file(test_m_path)

test_w_path =  os.path.join(current_directory, "data/test_w/*.jpg")
test_w_images = fn.read_data_from_file(test_w_path)


# Extracting filenames
train_images_files_names = list(train_images.keys())

# Extracting filenames for training
train_file_names = train_images_files_names[:93*6] # first 93 triplets
# Extracting filenames for validation
valid_file_names = train_images_files_names[93*6:] # remaining for validation. Roughly 80:20 ratio.
# Extracting filenames for test_m
test_m_file_names = list(test_m_images.keys())
# Extracting filenames for test_m
test_w_file_names = list(test_w_images.keys())

# Calculate the number of triplets for each set
train_num_of_triplets = len(train_file_names) // 6
valid_num_of_triplets = len(valid_file_names) // 6
test_m_num_of_triplets = len(test_m_file_names) // 6
test_w_num_of_triplets = len(test_w_file_names) // 6


train_data = fn.create_np_data_array(train_num_of_triplets, train_images, train_file_names)
valid_data = fn.create_np_data_array(valid_num_of_triplets, train_images, valid_file_names)
test_m_data = fn.create_np_data_array(test_m_num_of_triplets, test_m_images, test_m_file_names)
test_w_data = fn.create_np_data_array(test_w_num_of_triplets, test_w_images, test_w_file_names)


same_train_data = fn.generate_same_pair(train_data)
not_the_same_train_data = fn.generate_different_pair(train_data)


# Define the hyperparameter values to explore
learning_rates = [0.01, 0.001, 0.0001]
batch_sizes = [4, 16, 32]
epochs_values = [25, 40]

# Initialize variables to store the best results
best_accuracy = 0.0
best_hyperparameters = {}

toy_train_data = train_data[:16]

# Nested loop to iterate over hyperparameter values
for lr, batch_size, epochs in itertools.product(learning_rates, batch_sizes, epochs_values):
    print(f"Training with learning_rate={lr}, batch_size={batch_size}, epochs={epochs}")

    # Train the model multiple times (e.g., 3 times) and calculate mean accuracy
    mean_accuracies = []
    curr_best_acc = 0
    for _ in range(40):
        # Create and train the model using the specified hyperparameters
        CNNChannel_model = ML_DL_Functions3.CNNChannel()

        checkpoints_path = os.path.join(current_directory, "checkpoints_CNN")
        cnn_channel_training_curve = fn.train_model(CNNChannel_model, train_data , valid_data, learning_rate=lr, weight_decay=0.0005, batch_size=batch_size, epochs=epochs, checkpoint_path=checkpoints_path)
        # Count the total number of trainable parameters

        # Finding the index of the epoch with the highest validation accuracy. If there are multiple epochs with the same highest validation accuracy, the last one is chosen.
        best_epoch = len(cnn_channel_training_curve['validation_accuracies']) - 1 - cnn_channel_training_curve['validation_accuracies'][::-1].index(max(cnn_channel_training_curve['validation_accuracies']))

        # Load the best model and evaluate accuracy on test data
        best_model_path = f"{checkpoints_path}/model_epoch_{best_epoch}.pk"
        print("The best epoch we saved is: ", best_epoch+1 )
        best_CNNChannel_model = ML_DL_Functions3.CNNChannel()
        best_CNNChannel_model.load_state_dict(torch.load(best_model_path))
        best_CNNChannel_model.eval()
        w_cnn_channel_pos_acc, w_cnn_channel_neg_acc = fn.get_accuracy(best_CNNChannel_model, test_w_data, batch_size=batch_size)
        m_cnn_channel_pos_acc, m_cnn_channel_neg_acc = fn.get_accuracy(best_CNNChannel_model, test_m_data, batch_size=batch_size)
        print(f'Men CNNChannel Model Accuracy - Positive: {w_cnn_channel_pos_acc:.2%}, Negative: {w_cnn_channel_neg_acc:.2%}')
        print(f'Women CNNChannel Model Accuracy - Positive: {m_cnn_channel_pos_acc:.2%}, Negative: {m_cnn_channel_neg_acc:.2%}')
        

        # Calculate and store the mean accuracy
        mean_accuracy = max(cnn_channel_training_curve['validation_accuracies'])
        mean_accuracies.append(mean_accuracy)
        print(f' CNNChannel Model Validation Mean Accuracy{mean_accuracy:.2%}')
        if( mean_accuracy > curr_best_acc):
            curr_best_acc = mean_accuracy

            # Save the best model
            torch.save(best_CNNChannel_model.state_dict(), f"{checkpoints_path}/best_CNN_model.pk")
            print(f"Best Model saved at: {checkpoints_path}")
        

    # Calculate the mean of mean_accuracies
    mean_accuracy_value = sum(mean_accuracies) / len(mean_accuracies)
    print("Mean accuracy (for this set of parameters): ", mean_accuracy_value)
print('The best accuray is: ', curr_best_acc)

