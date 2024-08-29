import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv('genres.csv')

# Ensure the genre column is treated as a list of genres
df['genre'] = df['genre'].apply(lambda x: x.split(','))  # Assuming genres are comma-separated

# Group by 'id' and aggregate genres
df_grouped = df.groupby('id')['genre'].apply(lambda x: sum(x, [])).reset_index()

# Define the labels
labels = ['Comedy', 'Adventure', 'Thriller', 'Drama', 'Science Fiction', 'Action',
          'Music', 'Romance', 'History', 'Crime', 'Animation', 'Mystery', 'Horror',
          'Family', 'Fantasy', 'War', 'Western', 'TV Movie', 'Documentary']

# Initialize MultiLabelBinarizer with the defined labels
mlb = MultiLabelBinarizer(classes=labels)

# Fit and transform the aggregated genre lists to a binary matrix
binary_labels = mlb.fit_transform(df_grouped['genre'])

# Add the binary labels back to the DataFrame as individual columns
for i, label in enumerate(labels):
    df_grouped[label] = binary_labels[:, i]

# Save the updated DataFrame to a new CSV
df_grouped.to_csv('binary_genres_grouped.csv', index=False)




class FilmPosterDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # Load the labels CSV file
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Get the image file name
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        
        # Open the image
        image = Image.open(img_name).convert('RGB')
        
        # Get the binary labels
        labels = self.data_frame.iloc[idx, 1:].values.astype('float32')
        
        # Apply any transformations (e.g., resizing, normalization)
        if self.transform:
            image = self.transform(image)
        
        return image, labels

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match the input size expected by the model
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Create the dataset
dataset = FilmPosterDataset(csv_file='binary_genres_grouped.csv', root_dir='./posters/', transform=transform)

# Create a DataLoader for batching and shuffling the data
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example: iterating through the dataset
for images, labels in dataloader:
    print(images.shape, labels.shape)
    break  # Remove this line to loop through the entire dataset

