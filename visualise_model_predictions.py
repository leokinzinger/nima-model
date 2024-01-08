import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image
import os

def visualize_model_predictions(csv_file, folder_path,output_file):
    """
    Visualize model predictions by displaying images in quartiles based on their mean score predictions.

    Parameters:
    csv_file (str): Path to the CSV file containing image IDs and mean score predictions.
    folder_path (str): Path to the folder containing the images.

    """
    # Load the CSV data
    data = pd.read_csv(csv_file)
    
    # Compute quartiles
    data['quartile'] = pd.qcut(data['mean_score_prediction'], 4, labels=False)
    
    # Set up the matplotlib figure
    fig, axs = plt.subplots(4, 10, figsize=(20, 8))
    
    for quartile in range(4):
        # Get a subset of data for the current quartile
        quartile_data = data[data['quartile'] == quartile]
        
        # Randomly select 10 images (or all images if less than 10)
        selected_images = quartile_data.sample(min(len(quartile_data), 10))
        
        for i, (idx, row) in enumerate(selected_images.iterrows()):
            # Load and display the image
            image_path = os.path.join(folder_path, row['image_id'] + '.jpg')
            image = Image.open(image_path)
            axs[quartile, i].imshow(image)
            axs[quartile, i].set_title(f'Score: {row["mean_score_prediction"]:.2f}')
            axs[quartile, i].axis('off')
    
    plt.tight_layout()
    # Save the figure
    plt.savefig(output_file, bbox_inches='tight')

# Example usage
csv_file = 'Z:\\3_NIMA\\nima-model\\src\\tests\\2023_11_05_nima_results.csv'
folder_path = 'Z:\\Leo\\by_url'
output_file = 'Z:\\3_NIMA\\nima-model\\src\\tests\\visualised_results.png'
visualize_model_predictions(csv_file, folder_path,output_file)
