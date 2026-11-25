import numpy as np

folder = "deer_horse_data/noisy_datasets/"

clean = np.load(folder + "train_noise_0pct.npz")
noise2 = np.load(folder + "train_noise_2pct.npz")
noise5 = np.load(folder + "train_noise_5pct.npz")

# Check whether all datasets contain the same images
print("Same images (2%):",
      np.array_equal(clean["images"], noise2["images"]))

print("Same images (5%):",
      np.array_equal(clean["images"], noise5["images"]))

# Count incorrect labels
print("Wrong labels (2%):",
      np.sum(clean["labels"] != noise2["labels"]))

print("Wrong labels (5%):",
      np.sum(clean["labels"] != noise5["labels"]))