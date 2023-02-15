import os
from tqdm.auto import tqdm
import subprocess
import click
import tempfile
import string
import shutil
import h5py
import numpy as np
import imageio.v3 as imageio


def _get_symbols(mode, symbols=None):
    if mode == "letters":
        if symbols is not None:
            return [symbol.strip() for symbol in symbols.split(",")]
        else:
            return list(f"{string.ascii_lowercase}{string.ascii_uppercase}")
    elif mode == "numbers":
        if symbols is not None:
            return [symbol.strip() for symbol in symbols.split(",")]
        else:
            return list(string.digits)
    elif mode == "alphanumeric":
        return list(f"{string.ascii_lowercase}{string.ascii_uppercase}{string.digits}")


def _pack_images(output_file, symbol, generated_images_paths):
    with h5py.File(output_file, "a") as storage:
        labels = []
        vectors = []
        for filepath in generated_images_paths:
            try:
                fontname = filepath.split(os.path.sep)[-1][:-4]
                suffix = filepath.split(".")[-1]
                if suffix == "gif":
                    image = imageio.imread(filepath)[0, :, :, 1] # we work with BW images only
                elif suffix == "jpeg" or suffix == "jpg":
                    image = imageio.imread(filepath)
                elif suffix == "png":
                    image = imageio.imread(filepath)[:, :, 1]  # we work with BW images only
                labels.append(fontname)
                vectors.append(image)
            except Exception as e:
                print(f"Failed to process {filepath}: {e}")

        storage.create_dataset(f"{symbol}-labels", data=labels)
        storage.create_dataset(f"{symbol}-vectors", data=np.stack(vectors, axis=0))


def _close_symbol(fonts_root, generated_images_paths, output_file, symbol, images_out, tmp_dir):
    if generated_images_paths == []:
        print(f"No font files found in {fonts_root}")
        exit(1)

    _pack_images(output_file, symbol, generated_images_paths)
    if images_out is not None:
        shutil.move(tmp_dir, os.path.join(images_out, symbol))


@click.command()
@click.option("--fonts-root", required=False, default=".", help="Full path to root folder with font files")
@click.option("--raster-size", required=False, default="64x64", help="Size of produced images")
@click.option("--test-run", is_flag=True, default=False,
              help="When this flag is enabled one image is produced and then scirpt ends. Useful for checking correctness of paths and other parametrization.")
@click.option("--images-out", required=False,
              help="Directory, where generated images will be placed. When this is not defined, no images are produced h5py file.")
@click.option("--mode", required=False, default="letters",
              help="Defines what to generate. Choices are alphanumeric, numbers, letters.")
@click.option("--symbols", required=False, default=None,
              help="Parameter is used when --mode=letters|numbers to reduce set of letters/numbers to specified subset.")
@click.option("--image-output-format", default="gif", help="Format of rasterized images.")
@click.option("-o", "--output-file", default="fonts-data.h5", help="Output file with data.")
def generate_data(fonts_root, raster_size, test_run, images_out, mode, symbols, image_output_format, output_file):
    if images_out is not None and not os.path.isdir(images_out):
        print(f"Directory does not exist {images_out}. Please create it first.")
        exit(1)
    if image_output_format not in ["gif", "jpg", "jpeg", "png"]:
        print(f'Unsupported file format {image_output_format}. Select one of {["gif", "jpg", "jpeg", "png"]}')
        exit(1)

    generated_images_paths = []
    generated_symbols = _get_symbols(mode, symbols)
    for symbol in tqdm(generated_symbols, total=len(generated_symbols)):

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Go thru whole repository and find out ttf fonts
            for root, dirs, files in os.walk(fonts_root):
                for file in files:
                    if "ttf" == file[-3:] or "otf" == file[-3:]:
                        # print(f"[{symbol}]: {file}") # Debug
                        # hold the folder structure as the input has
                        os.makedirs(root.replace(fonts_root, tmp_dir), exist_ok=True)
                        out_path = os.path.join(root.replace(fonts_root, tmp_dir), f"{file[:-3]}{image_output_format}")
                        with subprocess.Popen(["convert",
                                               "-size", f"{raster_size}",
                                               "-background", "white",
                                               "-font", os.path.join(root, file),
                                               "-fill", "black",
                                               "-gravity", "Center",
                                               f'label:{symbol}',
                                               "-flatten", out_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
                            # print(proc.stdout.read()) # Debug
                            # print(proc.stderr.read()) # Debug
                            pass

                        generated_images_paths.append(out_path)

                        if test_run:
                            _close_symbol(fonts_root, generated_images_paths, output_file, symbol, images_out, tmp_dir)
                            exit(0)

            _close_symbol(fonts_root, generated_images_paths, output_file, symbol, images_out, tmp_dir)


if __name__ == "__main__":
    generate_data()
