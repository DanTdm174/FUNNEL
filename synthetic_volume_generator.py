import numpy as np

# Generating a 64^3 array using np.random.rand
array_3d = np.random.rand(64, 64, 64)

#Smoothing it by importing gaussian filter; Applying to each axis; Using a factor of 1

from scipy.ndimage import gaussian_filter

smoothed_array = gaussian_filter(array_3d, sigma=1.0)

print(smoothed_array.shape)
print(smoothed_array.mean(), smoothed_array.std())

import nibabel as nib

affine = np.eye(4)
img = nib.Nifti1Image(smoothed_array, affine)
img.to_filename("test_volume.nii.gz")

loaded = nib.load("test_volume.nii.gz")
loaded_array = loaded.get_fdata()

print(loaded_array.shape)
print(np.allclose(smoothed_array, loaded_array))
# Creating the lesion mask (coords have the x, y, and z coords of the voxel)
def add_lesion(array, center, radius, intensity_boost):
    coords = np.indices((64, 64, 64))

    # Creating the 64x64x64 using "distance" and the np.sqrt formula
    cx, cy, cz = center
    distance = np.sqrt((coords[0] - cx)**2 + (coords[1] - cy)**2 + (coords[2] - cz)**2)

    #Actually building the mask using boolean arrays
    mask = distance < radius

    # Applying it by creating lesioned_array and copying the smoothed_array.copy
    result = array.copy()
    result[mask] += intensity_boost
    
    return result

lesioned_array = add_lesion(smoothed_array, center=(32, 32, 32), radius=8, intensity_boost=0.3)

print(lesioned_array.mean(), smoothed_array.mean())  # lesioned should be slightly higher

# Importing matplotlib as plt

import matplotlib.pyplot as plt

plt.imshow(lesioned_array[32, :, :], cmap='gray')
plt.colorbar()
plt.show()

#Lesion detecting mechanism
def generate_sample(has_lesion):
    array3d = np.random.rand(64, 64, 64)
    smoothedarray = gaussian_filter(array3d, sigma=1.0)
    
    if has_lesion:
        random_center = (
            np.random.randint(16, 48),
            np.random.randint(16, 48),
            np.random.randint(16, 48)
        )
        random_radius = np.random.uniform(4.0, 10.0)
        
        final_array = add_lesion(smoothedarray, center=random_center, radius=random_radius, intensity_boost=0.3)
        label = 1
    else:
        final_array = smoothedarray
        label = 0
    
    return final_array, label

healthy_array, healthy_label = generate_sample(has_lesion=False)
lesion_array, lesion_label = generate_sample(has_lesion=True)

array1, label1 = generate_sample(has_lesion=True)
array2, label2 = generate_sample(has_lesion=True)
array3, label3 = generate_sample(has_lesion=True)

print(array1.mean(), array2.mean(), array3.mean())

print(healthy_label, lesion_label)
print(healthy_array.mean(), lesion_array.mean())
# Creating the looping mechanism to save to a CSV file
# inside loop:
import csv
import os

output_dir = "synthetic_data"
os.makedirs(output_dir, exist_ok=True)

manifest = []

for i in range(30):
    has_lesion_flag = np.random.choice([True, False])
    array, label = generate_sample(has_lesion_flag)
    
    filename = os.path.join(output_dir, f"sample_{i}.nii.gz")
    affine = np.eye(4)
    img = nib.Nifti1Image(array, affine)
    img.to_filename(filename)
    
    manifest.append((filename, label))

manifest_path = os.path.join(output_dir, "manifest.csv")
with open(manifest_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(manifest)
    
from torch.utils.data import Dataset
import torch

class SyntheticBrainDataset(Dataset):
    def __init__(self, manifest_path):
        self.data = []
        with open(manifest_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # skip header row
            for row in reader:
                self.data.append((row[0], int(row[1])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, label = self.data[idx]
        
        loaded = nib.load(filename)
        loaded_array = loaded.get_fdata()
        
        tensor = torch.tensor(loaded_array, dtype=torch.float32)
        
        return tensor, label

dataset = SyntheticBrainDataset("synthetic_data/manifest.csv")

sample_tensor, sample_label = dataset[0]
print(sample_tensor.shape, sample_label) 

from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch_tensors, batch_labels in dataloader:
    print(batch_tensors.shape, batch_labels)
    break  