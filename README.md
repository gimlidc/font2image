# Font2Image

Purpose of this tool is to create dataset useful for training AI for classification of set of fonts. The source here is the collection of google fonts (but simply other can be added, please check their LICENSE first) in vector format. Vector format is converted into raster format and finally these rasters are loaded into a numpy array suitable for Pythonic training.

## Requirements

- ImageMagic (the convert feature for font rendering) [installation instructions](https://imagemagick.org/script/download.php)
- poetry ^1.1.2 [installation instructions](https://python-poetry.org/docs/#installation)

## Run

```bash
poetry install
python scripts/font2h5matrix.py --fonts-root fonts --mode letters --symbols a --image-output-format jpeg -o letter-a-as-jpeg.h5
```

## Roadmap (TBD)

- dockerization for comaptibility

