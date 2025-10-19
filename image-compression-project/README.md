# Image Compression Project

This project is designed to compress TIFF images into JPG format while preserving the original directory structure. It also handles the corresponding JSON files, ensuring they are saved in the new directory structure.

## Project Structure

```
image-compression-project
├── src
│   ├── main.py                # Entry point of the application
│   ├── image_processor.py      # Contains ImageProcessor class for image operations
│   ├── json_processor.py       # Contains JsonProcessor class for JSON operations
│   └── utils
│       ├── __init__.py        # Initializes the utils package
│       ├── file_utils.py       # Utility functions for file operations
│       └── config.py           # Configuration settings for the project
├── scripts
│   └── compress_images.py      # Script to execute the image compression process
├── requirements.txt            # Lists project dependencies
├── config.json                 # Configuration settings for input and output paths
└── README.md                   # Documentation for the project
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd image-compression-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Configure the input and output paths in `config.json`.

2. Run the compression script:
   ```
   python scripts/compress_images.py
   ```

This will compress all TIFF images in the specified input directory and save them as JPG files in the corresponding output directory, along with the modified JSON files.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.