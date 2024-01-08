import os
import pandas as pd
import shutil

# Path to the CSV file
csv_file_path = 'Z:\\3_NIMA\\nima-model\\src\\tests\\2023_11_05_nima_results.csv'
# Path to the folder with images
images_folder_path = 'Z:\\Leo\\by_url'
# Path to the folder where missing images will be copied
missing_images_folder_path = 'Z:\\3_NIMA\\nima-model\\src\\tests\\missing_images'

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Remove duplicate entries based on 'image_id'
df.drop_duplicates(subset='image_id', keep='first', inplace=True)

# Save the dataframe to the same CSV file path
df.to_csv(csv_file_path, index=False)

# Extract image ids from the dataframe and add the .jpg extension
image_ids = set(df['image_id'] + '.jpg')

# List all image files in the target folder
image_files = set(os.listdir(images_folder_path))

# Find the difference - images that are in the folder but not in the CSV
missing_images = image_files - image_ids

# Check if the folder for missing images exists, if not, create it
if not os.path.exists(missing_images_folder_path):
    os.makedirs(missing_images_folder_path)

# Copy the missing images
for image in missing_images:
    source = os.path.join(images_folder_path, image)
    destination = os.path.join(missing_images_folder_path, image)
    shutil.copy2(source, destination)
    print(f"Copied {image} to {missing_images_folder_path}")

print("Operation completed.")
