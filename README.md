# SAR Ship Classification with Super-Resolution

This repository contains code for enhancing SAR ship classification through super-resolution techniques. We compare traditional interpolation methods with deep learning-based super-resolution (SR) models and evaluate their impact on image quality and classification performance using VGG and MSVGG models.

## Repository Structure

The repository consists of three main files:

1. **`code.py`**  
   This script trains the ship classification models using super-resolved images. It contains the implementation of two models:
   - **VGG Model**: A basic VGG network for classification.
   - **MSVGG Model**: A multi-scale VGG network designed for improved classification performance.
   
   The script handles training, validation, and evaluation of these models on super-resolved images.

2. **`resolution.py`**  
   This script applies various super-resolution techniques to low-resolution images. It includes:
   - **Deep learning models** (EDSR, RCAN, CARN) for super-resolution.
   - **Traditional interpolation methods** (bilinear, bicubic, Lanczos, nearest-neighbor).
   
   The output is super-resolved images, saved for further use in training and testing the classification models.

3. **`image_quality.py`**  
   This script calculates image quality metrics to evaluate the super-resolved images. It computes:
   - **Peak Signal-to-Noise Ratio (PSNR)**
   - **Structural Similarity Index (SSIM)**  
   
   Quality is assessed for both 2x and 4x resolution images, providing a comparison of the performance of different SR methods.

## Super-Resolution Methods

The repository includes the following super-resolution methods:
- **Traditional Interpolation**: Bilinear, Bicubic, Lanczos, Nearest-Neighbor.
- **Deep Learning Models**: EDSR, RCAN, CARN.

## Image Quality Metrics

The `image_quality.py` script measures the performance of SR methods using:
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the ratio between the maximum possible signal and the noise affecting the image.
- **SSIM (Structural Similarity Index)**: Evaluates the perceived quality of images by comparing structural information.

## Installation

To run the code, clone this repository and install the required dependencies:

```bash
git clone https://github.com/cm-awais/SAR_Super_resolution.git
cd SAR_Super_resolution
pip install -r requirements.txt
```

## Usage

1. **Super-Resolution**  
   To perform super-resolution on images, use the `resolution.py` script:

   ```bash
   python resolution.py --input_dir /path/to/images --output_dir /path/to/save --method edsr
   ```

   Replace `edsr` with the desired method (e.g., `bilinear`, `bicubic`, `lanczos`, `nearest-neighbor`, `rcan`, `carn`).

2. **Train Models**  
   Use `code.py` to train the models (VGG or MSVGG) on the super-resolved images:

   ```bash
   python code.py --train_dir /path/to/super-resolved-images --model vgg
   ```

   Replace `vgg` with `msvgg` to use the multi-scale VGG model.

3. **Image Quality Evaluation**  
   To evaluate image quality, use `image_quality.py`:

   ```bash
   python image_quality.py --input_dir /path/to/super-resolved-images --scale 2x
   ```

   Adjust `--scale` for 2x or 4x super-resolution comparisons.

## Results

The study highlights that improvements in image quality (PSNR, SSIM) do not always correlate with better classification performance. Traditional methods such as Lanczos may outperform deep learning models in some cases, particularly when applied to a multi-scale VGG (MSVGG) model.

## Contributing

Feel free to open issues or pull requests if you encounter any problems or have suggestions for improvements.

---
