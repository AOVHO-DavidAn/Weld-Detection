from image_processor import ImageProcessor
from json_processor import JsonProcessor
import os
import json

def main():
    # Load configuration
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    input_image_dir = config['input_image_dir']
    output_image_dir = config['output_image_dir']
    input_json_dir = config['input_json_dir']
    output_json_dir = config['output_json_dir']

    # Create output directories if they don't exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)

    # Initialize processors
    image_processor = ImageProcessor(input_image_dir, output_image_dir)
    json_processor = JsonProcessor(input_json_dir, output_json_dir)

    # Process images
    image_processor.compress_images()

    # Process JSON files
    json_processor.process_json_files()

if __name__ == "__main__":
    main()