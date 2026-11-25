import os
import numpy as np

# The clean dataset was made by prepare_deer_horse.py
project_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(project_folder, "deer_horse_data")
output_folder = os.path.join(data_folder, "noisy_datasets")
os.makedirs(output_folder, exist_ok=True)

# 0% is our original data. 2% and 5% are label-noise experiments.
noise_levels = [0, 2, 5]
noise_seed = 1234

# Load our existing 10:1 training dataset
train = np.load(os.path.join(data_folder, "train.npz"))
images = train["images"]
clean_labels = train["labels"]
original_indices = train["original_indices"]

# Find positions within this new training dataset
# Deer = 0, Horse = 1
deer_positions = np.where(clean_labels == 0)[0]
horse_positions = np.where(clean_labels == 1)[0]

# Shuffle once and reuse the order for every noise level
# This makes the selected noisy examples reproducible.
rng = np.random.default_rng(noise_seed)
rng.shuffle(deer_positions)
rng.shuffle(horse_positions)

for noise_percent in noise_levels:
    noisy_labels = clean_labels.copy()  # do not change the original labels

    # Number of labels to change (out of the full training dataset)
    total_to_change = round(len(clean_labels) * noise_percent / 100)

    # Change the same number in each direction to keep 4000:400
    if total_to_change % 2 != 0:
        raise ValueError("This noise percentage needs an even number of changes")

    per_class = total_to_change // 2
    if per_class > len(horse_positions):
        raise ValueError("Noise is too high to preserve class counts")

    selected_deer = deer_positions[:per_class]
    selected_horse = horse_positions[:per_class]

    noisy_labels[selected_deer] = 1  # Deer incorrectly labelled Horse
    noisy_labels[selected_horse] = 0  # Horse incorrectly labelled Deer

    # True where a label has been deliberately changed
    noisy_mask = noisy_labels != clean_labels

    file_name = "train_noise_" + str(noise_percent) + "pct.npz"
    np.savez_compressed(
        os.path.join(output_folder, file_name),
        images=images,
        labels=noisy_labels,
        clean_labels=clean_labels,
        noisy_mask=noisy_mask,
        original_indices=original_indices,
        noise_percent=np.array(noise_percent),
        noise_seed=np.array(noise_seed),
    )

    print(file_name)
    print("  Wrong labels:", np.sum(noisy_mask))
    print("  Deer labels:", np.sum(noisy_labels == 0))
    print("  Horse labels:", np.sum(noisy_labels == 1))

print("Finished! Validation and test datasets have not been changed.")
