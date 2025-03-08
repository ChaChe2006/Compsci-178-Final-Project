import os
import pandas as pd

directory = '../countries' # path to directory with the csv files
dataframes = []

# List of all csv files in the directory
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Sort based on numeric value of file names
csv_files.sort(key=lambda x: int(x.split('_')[4])) #extract the first number in the file name

# Loop through all CSV files in the directory
for filename in csv_files:
    file_path = os.path.join(directory, filename)

    # Keep header line only if this is the first file
    if len(dataframes) == 0:
        datafile = pd.read_csv(file_path)
    else:
        datafile = pd.read_csv(file_path, header = 0, names=['latitude', 'longitude', 'country'])

    dataframes.append(datafile)

# Combine dataframes, ignoring indexes of individual files
combined_datafile = pd.concat(dataframes, ignore_index = True)

# Save combined dataframes to a new csv without adding an index column
combined_datafile.to_csv('coordinates_with_country.csv', index=False)