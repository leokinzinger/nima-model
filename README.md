# NIMA Model Evaluation

This repository contains the NIMA model evaluation script, designed to assess image quality using the NIMA (Neural Image Assessment) model. The script allows for the evaluation of images using a pre-trained MobileNet model.

## Installation

Before running the script, ensure that you have Python installed on your system. Clone the repository to your local machine.

## Usage

To evaluate images using the NIMA model, use the following command:

```bash
python src/evaluater/predict.py --base-model-name <model_name> --weights-file <path_to_weights_file> --image-source <path_to_image_directory> --predictions-file <path_to_output_csv>
```

Replace the placeholders with the appropriate paths and filenames:

- <model_name>: Name of the base model to use, e.g., MobileNet.
- <path_to_weights_file>: Path to the pre-trained model weights file (HDF5 format), relative to the root of the project. I have included the pretrained file from the original project: models/MobileNet/weights_mobilenet_technical_0.11.hdf5
- <path_to_image_directory>: Directory containing the images to be evaluated, relative to the root of the project.
- <path_to_output_csv>: Path where the output CSV file containing predictions will be saved, relative to the root of the project.

# Example Command

```bash
python src/evaluater/predict.py --base-model-name MobileNet --weights-file "models/MobileNet/weights_mobilenet_technical_0.11.hdf5" --image-source "src/tests/test_images" --predictions-file "results/test_results.csv"
```

# Output
The script will process the images in the specified directory and output the results in a CSV file. This file will include the image names and their corresponding quality scores.

