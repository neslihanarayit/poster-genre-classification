import zipfile
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
import shutil

def unzip_folder(zip_path, extract_to):
    # Ensure the output directory exists
    os.makedirs(extract_to, exist_ok=True)

    # Open the zip file and extract all contents
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted '{zip_path}' to '{extract_to}'")

# Example usage
zip_path = 'archive.zip'  # Replace with the path to your zip file
extract_to = 'PosterGenreData'  # Replace with the desired output folder

# Run only for the first time

def split_dataset(dataset_path, train_dir, test_dir, test_size=0.3, random_seed=42):
    torch.manual_seed(random_seed)

    # Load the dataset
    dataset = ImageFolder(dataset_path)

    # Calculate train and test sizes
    total_size = len(dataset)
    test_size = int(total_size * test_size)
    train_size = total_size - test_size

    # Perform the split
    train_data, test_data = random_split(dataset, [train_size, test_size])

    # Helper to copy images to directories
    def copy_files(data, target_dir):
        for idx, (img_obj, label) in enumerate(data):
            # Get the original path using dataset.samples
            img_path, _ = dataset.samples[idx]

            # Get the genre name from the label
            genre_name = dataset.classes[label]
            target_genre_dir = os.path.join(target_dir, genre_name)
            os.makedirs(target_genre_dir, exist_ok=True)

            # Copy the image to the appropriate folder
            shutil.copy(img_path, target_genre_dir)

    print("Copying train images...")
    copy_files(train_data, train_dir)

    print("Copying test images...")
    copy_files(test_data, test_dir)

    print("Train/Test split complete.")

# Example usage
dataset_path = 'Dataset/Posters/Posters_Train'
new_train_folder = 'Dataset/Posters/New_Train'
new_test_folder = 'Dataset/Posters/New_Test'

# Perform the split with 70/30 ratio
# split_dataset(dataset_path, new_train_folder, new_test_folder, test_size=0.3)


def getting_percentage_of_dominant_colors(cluster, centroids):
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins=labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # iterate through each cluster's color and percentage
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    for (percent, color) in colors:
        try:
            if percent > 0.50:
                print(color, "{:0.2f}%".format(percent * 100))
                return True
        except Exception as e:
            print(str(e))
    return False





def count_images_in_folders(base_dir):
    """
    Recursively counts the number of images in each genre folder inside the given base directory.
    Returns a dictionary with folder names and their corresponding image counts.
    """
    image_counts = {}
    total_images = 0

    # Walk through all subdirectories of the base directory
    for root, dirs, files in os.walk(base_dir):
        if len(files) > 0:  # If the folder contains files
            genre_name = os.path.basename(root)  # Extract the genre name
            image_counts[genre_name] = len(files)  # Count the number of files
            total_images += len(files)

    return image_counts, total_images

def plot_histogram(image_counts, title):
    """
    Plots a histogram using the image counts.
    """
    genres = list(image_counts.keys())
    counts = list(image_counts.values())

    plt.figure(figsize=(15, 5))  # Adjust the figure size for readability
    plt.bar(genres, counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')  # Rotate genre labels for better visibility
    plt.xlabel('Genres')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.tight_layout()  # Ensure everything fits
    plt.show()

# Paths to the dataset folders
train_folder = 'Dataset/Posters/Train'
test_folder = 'Dataset/Posters/Test'
validation_folder = 'Dataset/Posters/Validation'

# Count images in train and validation folders
train_counts, train_total = count_images_in_folders(train_folder)
test_counts, test_total = count_images_in_folders(test_folder)
validation_counts, valid_total = count_images_in_folders(validation_folder)

# Plot histograms
print("Train Image Counts:", train_counts, "Total: ", train_total)
plot_histogram(train_counts, 'Number of Images per Genre (Training Set)')

print("test Image Counts:", test_counts, "Total: ", test_total)

plot_histogram(test_counts, 'Number of Images per Genre (Test Set)')

print("Validation Image Counts:", validation_counts)
plot_histogram(validation_counts, 'Number of Images per Genre (Validation Set)')


'''

plt.imshow(image)
plt.axis('off')  # Turn off axis for cleaner visualization
plt.title(f"Genre: {train_classes[label]}")
plt.show()

# Path to the train dataset (adjust based on your setup)
dataset_path = "Dataset/Posters/Posters_Train"


def get_random_image_with_label():
    """Select a random image and return it with its genre label."""
    genre_folder = random.choice(os.listdir(dataset_path))
    images_path = os.path.join(dataset_path, genre_folder)
    image_file = random.choice(os.listdir(images_path))
    label = genre_folder  # Use the folder name as the label
    image = Image.open(os.path.join(images_path, image_file)).convert('RGB')
    return image, label


def visualize_images_with_transform(n=5):
    """Visualize n random images before and after transformations."""
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))

    for i in range(n):
        # Get a random image and label
        original_image, label = get_random_image_with_label()

        # Apply the transformation
        transformed_image = test_transform(original_image)

        # Convert transformed Tensor back to PIL for visualization
        transformed_image = transforms.ToPILImage()(transformed_image)

        # Plot original and transformed images
        axes[i, 0].imshow(original_image)
        axes[i, 0].set_title(f"Original: {label}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(transformed_image)
        axes[i, 1].set_title(f"Transformed: {label}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


# Visualize 5 random images with transformations
# visualize_images_with_transform(n=5)



# Display dataset info

get_dataset_info(train_dataset, "Train Dataset")
get_dataset_info(test_dataset, "Test Dataset")
get_dataset_info(validation_dataset, "Validation Dataset")'''
