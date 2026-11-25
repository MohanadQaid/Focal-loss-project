
import os
import pickle
import numpy as np

# Folder containing the CIFAR-10 data files
project_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(project_folder, "cifar-10-batches-py")
output_folder = os.path.join(project_folder, "deer_horse_data")

os.makedirs(output_folder, exist_ok=True)


#load a CIFAR-10 batch
def load_batch(file_name):
    file_path = os.path.join(data_folder, file_name)

    with open(file_path, "rb") as file:
        batch = pickle.load(file, encoding="bytes")

    images = batch[b"data"].reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)
    labels = np.array(batch[b"labels"])

    return images, labels


# Load the five training batches
all_images = []
all_labels = []

for i in range(1, 6):
    images, labels = load_batch("data_batch_" + str(i))
    all_images.append(images)
    all_labels.append(labels)

train_images = np.concatenate(all_images)
train_labels = np.concatenate(all_labels)

test_images, test_labels = load_batch("test_batch")


# Select deer (class 4) and horse (class 7)
deer_indices = np.where(train_labels == 4)[0]
horse_indices = np.where(train_labels == 7)[0]


# Shuffle the data using a fixed random seed
random = np.random.default_rng(2026)

random.shuffle(deer_indices)
random.shuffle(horse_indices)


# Select 1000 deer and 1000 horse for validation
validation_indices = np.concatenate((
    deer_indices[:1000],
    horse_indices[:1000]
))


# Select 4000 deer and 400 horse for training
training_indices = np.concatenate((
    deer_indices[1000:],
    horse_indices[1000:1400]
))

random.shuffle(training_indices)
random.shuffle(validation_indices)


# Select deer and horse from the official test set
testing_indices = np.where(
    (test_labels == 4) | (test_labels == 7)
)[0]


# Function to save the new dataset
def save_dataset(name, images, labels, indices):

    selected_images = images[indices]

    # Change labels: Deer = 0, Horse = 1
    selected_labels = (labels[indices] == 7).astype(np.uint8)

    np.savez_compressed(
        os.path.join(output_folder, name + ".npz"),
        images=selected_images,
        labels=selected_labels,
        original_indices=indices
    )

    deer_count = np.sum(selected_labels == 0)
    horse_count = np.sum(selected_labels == 1)

    print(name, "- Deer:", deer_count, "Horse:", horse_count)


# Save training, validation and testing datasets
save_dataset("train", train_images, train_labels, training_indices)
save_dataset("validation", train_images, train_labels, validation_indices)
save_dataset("test", test_images, test_labels, testing_indices)


# Save the selected indices
np.savez_compressed(
    os.path.join(output_folder, "split_indices.npz"),
    train_indices=training_indices,
    validation_indices=validation_indices,
    test_indices=testing_indices,
    selection_seed=np.array(2026)
)

print("Finished! Files saved in deer_horse_data")
