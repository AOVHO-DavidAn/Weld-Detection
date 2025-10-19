class ImageProcessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def load_tiff_images(self):
        from pathlib import Path
        from PIL import Image

        tiff_images = list(Path(self.input_dir).rglob('*.tif'))
        return tiff_images

    def compress_image(self, image_path):
        img = Image.open(image_path)
        jpg_path = image_path.with_suffix('.jpg')
        return img, jpg_path

    def save_image(self, img, jpg_path):
        img.save(jpg_path, 'JPEG')

    def process_images(self):
        tiff_images = self.load_tiff_images()
        for tiff_image in tiff_images:
            img, jpg_path = self.compress_image(tiff_image)
            output_path = Path(self.output_dir) / jpg_path.relative_to(self.input_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_image(img, output_path)