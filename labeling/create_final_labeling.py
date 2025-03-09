import pandas as pd
import os

input_csv = 'coordinates_with_country.csv'
output_csv = 'image_paths_with_country.csv'
image_dir = 'E:\\UCI\\CS 178\\Project\\archive\\Streetview_Image_Dataset_Kaggle_25k' # Remember to replace with path of your local directory

# Read the intermediate CSV file
df = pd.read_csv(input_csv)

# Generate image paths based on row index
df['image_path'] = df.index.map(lambda x: os.path.join(image_dir, f'{x}.png'))

# Create a new dataframe
new_df = df[['image_path', 'country']]

# Save the new csv file
new_df.to_csv(output_csv, index=False)