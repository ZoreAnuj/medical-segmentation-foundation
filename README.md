# Medical Segmentation Foundation

This project explores a generalist foundation model for open-world medical image segmentation, based on the MedSegX research. It aims to provide a flexible framework for segmenting diverse anatomical structures and pathologies across various imaging modalities without task-specific retraining.

## Key Features
* Implements a vision-language foundation model for zero-shot and open-set medical segmentation.
* Supports inference on multiple imaging modalities (CT, MRI, X-ray, etc.).
* Includes utilities for processing and evaluating medical imaging datasets.

## Tech Stack
* Python
* PyTorch
* MONAI
* Transformers

## Getting Started
1. Clone the repository: `git clone https://github.com/zoreanuj/medical-segmentation-foundation.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run a sample inference script to test the model.