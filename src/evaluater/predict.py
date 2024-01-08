import os
import sys
from pathlib import Path

# Add the parent directory of `utils` to sys.path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
import glob
from PIL import Image
import pandas as pd
import json
import argparse
from utils.utils import calc_mean_score, save_json
from handlers.model_builder import Nima
from handlers.data_generator import TestDataGenerator

def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]

    return img_dir, [{'image_id': img_id}]

def image_dir_to_json(img_dir, img_type='jpg', batch_size=10, start_after=None):
    img_paths = glob.glob(os.path.join(img_dir, '*.' + img_type))
    img_paths.sort()  # Make sure the files are in a consistent order

    # Find the index of the starting image if a start_after is specified
    start_index = 0
    if start_after is not None:
        try:
            start_index = next(i for i, img_path in enumerate(img_paths) if start_after in img_path)
        except StopIteration:
            raise ValueError(f"Start after image {start_after} not found in the directory.")

    for i in range(start_index, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]
        samples = []
        for img_path in batch_paths:
            try:
                with Image.open(img_path) as img:
                    img.verify()  # This will check if the file is an image and not corrupt
                img_id = os.path.basename(img_path).split('.')[0]
                samples.append({'image_id': img_id})
                print(f"Loading: {img_id}")
            except (IOError, SyntaxError) as e:
                print(f'Bad file, could not process {img_path}: {e}')
        yield samples


def predict(model, data_generator):
    return model.predict_generator(data_generator, workers=8, use_multiprocessing=True, verbose=1)

def process_batch(nima, image_dir, samples, predictions_file, img_format):
    try:
        df = pd.DataFrame(samples)
        data_generator = TestDataGenerator(samples, image_dir, 64, 10, nima.preprocessing_function(), img_format=img_format)
        predictions = predict(nima.nima_model, data_generator)
        df['mean_score_prediction'] = [calc_mean_score(pred) for pred in predictions]

        if predictions_file is not None:
            # File path
            file_path = predictions_file

            try:
                # Check if the file exists
                if os.path.exists(file_path):
                    # If it exists, read the existing data
                    df_existing = pd.read_csv(file_path)
                    # Concatenate the new data with the existing data
                    df_concatenated = pd.concat([df_existing, df], ignore_index=True)
                    # Save the concatenated data back to the file
                    df_concatenated.to_csv(file_path, index=False)
                else:
                    # If the file doesn't exist, save the new data to the file
                    df.to_csv(file_path, index=False)
            except Exception as e:
                print(f"Error saving predictions to file: {e}")
    except Exception as e:
        print(f"Error processing batch: {e}")


def main(base_model_name, weights_file, image_source, predictions_file, img_format='jpg',start_after=None):
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)
    
    if os.path.isfile(image_source):
        image_dir, samples = image_file_to_json(image_source)
        process_batch(nima, image_dir, samples, predictions_file, img_format)
    else:
        image_dir = image_source
        for samples in image_dir_to_json(image_dir, img_type=img_format,start_after=start_after):
            process_batch(nima, image_dir, samples, predictions_file, img_format)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model-name', help='CNN base model name', required=True)
    parser.add_argument('-w', '--weights-file', help='path of weights file', required=True)
    parser.add_argument('-is', '--image-source', help='image directory or file', required=True)
    parser.add_argument('-pf', '--predictions-file', help='file with predictions', required=False, default=None)
    parser.add_argument('-sa', '--start-after', help='Start processing from the image file named after this argument', required=False, default=None)

    args = parser.parse_args()

    main(**args.__dict__)
