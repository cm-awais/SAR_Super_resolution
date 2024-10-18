# SAR Ship Classification with Super-Resolution

This repository focuses on applying and evaluating super-resolution techniques for Synthetic Aperture Radar (SAR) ship classification. Both traditional interpolation methods and deep learning-based super-resolution models are used, and their impact on image quality and classification performance is explored.

## Repository Structure

The repository consists of the following key files:

1. **`sar_super_resolution.ipynb`**  
   A Jupyter notebook that performs super-resolution using traditional interpolation methods (bilinear, bicubic, Lanczos, nearest-neighbor). It also calculates image quality metrics such as PSNR and SSIM, and saves the super-resolved images for further use in classification.

2. **`dl_res.py`**  
   This script performs super-resolution using deep learning-based models (EDSR, RCAN, CARN). It calculates image quality scores (PSNR, SSIM) and stores the super-resolved images for later use in the classification tasks.

3. **`test_resolution.py`**  
   This script evaluates the classification performance of two deep learning models (VGG and MSVGG) using the super-resolved images. It calculates accuracy, precision, recall, and F1 score to assess model performance on upscaled images.

## Super-Resolution Methods

The repository includes the following super-resolution methods:
- **Traditional Interpolation**: Bilinear, Bicubic, Lanczos, Nearest-Neighbor (implemented in `sar_super_resolution.ipynb`).
- **Deep Learning Models**: EDSR, RCAN, CARN (implemented in `dl_res.py`).

## Image Quality Metrics

The following metrics are used to evaluate the quality of super-resolved images:
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the clarity of the image by comparing the maximum possible signal to the noise affecting the image.
- **SSIM (Structural Similarity Index Measure)**: Assesses the perceived image quality by comparing structural information between the original and super-resolved images.

## Installation

To run the code, clone the repository and install the required dependencies:

```bash
git clone https://github.com/cm-awais/SAR_Super_resolution.git
cd SAR_Super_resolution
pip install -r requirements.txt
```

## Usage

1. **Super-Resolution with Interpolation**  
   Run the Jupyter notebook `sar_super_resolution.ipynb` to apply interpolation-based super-resolution and calculate image quality scores:

   ```bash
   jupyter notebook sar_super_resolution.ipynb
   ```

2. **Super-Resolution with Deep Learning**  
   Run `dl_res.py` to perform super-resolution using deep learning models (EDSR, RCAN, CARN) and calculate image quality scores:

   ```bash
   python dl_res.py
   ```

3. **Classification on Super-Resolved Images**  
   Run `test_resolution.py` to evaluate the classification results on the super-resolved images using VGG and MSVGG models:

   ```bash
   python test_resolution.py
   ```

## Results

This project highlights that while deep learning-based super-resolution models generally improve image quality, they don’t always guarantee better classification accuracy. Traditional interpolation methods may sometimes outperform deep learning models in specific scenarios, especially when using a multi-scale VGG (MSVGG) model.

## Contributing

Feel free to open issues or pull requests if you encounter any problems or have suggestions for improvements.
