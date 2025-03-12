import pandas as pd
import os

# Define file paths
input_csv = 'labeling/coordinates_with_country.csv'
output_csv = 'dataset/split data/coords_with_country_and_imagepath.csv'
train_csv = 'dataset/split data/train_coords_with_country.csv'
val_csv = 'dataset/split data/val_coords_with_country.csv'
test_csv = 'dataset/split data/test_coords_with_country.csv'

# Define image directory
image_dir = r"C:\Users\AYNum\OneDrive - personalmicrosoftsoftware.uci.edu\Documents\School\Year 2\Winter Quarter\CS 178\Prj\Streetview_Image_Dataset"

# Read CSV file
df = pd.read_csv(input_csv)

# Add 'image_path' column using row index
df['image_path'] = df.index.map(lambda x: os.path.join(image_dir, f'{x}.png'))

# Save new DataFrame with image paths
df.to_csv(output_csv, index=False)

# Remove countries with fewer than 20 entries
country_counts = df['country'].value_counts()
valid_countries = country_counts[country_counts >= 20].index
filtered_df = df[df['country'].isin(valid_countries)]

# Function to split data by country
def split_data(group):
    train_size = int(0.64 * len(group))
    val_size = int(0.16 * len(group))
    
    train = group.iloc[:train_size]
    val = group.iloc[train_size:train_size + val_size]
    test = group.iloc[train_size + val_size:]
    
    return train, val, test

# Split data
train_list, val_list, test_list = [], [], []
for _, group in filtered_df.groupby('country'):
    train, val, test = split_data(group)
    train_list.append(train)
    val_list.append(val)
    test_list.append(test)

# Concatenate results
train_df = pd.concat(train_list)
val_df = pd.concat(val_list)
test_df = pd.concat(test_list)

# Save split datasets
train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)
