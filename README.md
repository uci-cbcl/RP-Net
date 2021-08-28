[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# Recurrent Mask Refinement for Few-Shot Medical Image Segmentation

# Steps
1. Install any missing packages using pip or conda

2. Preprocess each dataset using utils/preprocess

3. Run notebooks/prepare_data_for_few_shot_learning.ipynb to generate files containing the range of each organ

# Train

# Test
Configure the yaml. An example is at yamls/example.yml. Change ckpt to trained model checkpoint
`python test.py test --yaml $PATH_TO_YAML`

