# doc_safety

a project for the machine learning course. 

fmph uniba, winter semester, 2025/2026.

this repository contains code and resources used to evaluate the safety of image enhancement models (drunet, dncnn, pix2pix) when used on documents with texts.

## repository structure

- [01_dataset](01_dataset): contains scripts to create the dataset used for evaluation
- [02_networks](02_networks): contains scripts to train the image enhancement models (drunet, dncnn, pix2pix), and training logs
- [03_evaluation](03_evaluation): contains scripts to evaluate the safety of the image enhancement models on the dataset, and the results obtained (IQA, OCR, plots, confusion matrices)
- [examples](examples): contains example images from the dataset, stains, textures, inference results, etc

other resources, that is 
- trained models
- (full) train, val and test datasets
- (full) textures and stains used for creating the dataset
- (full) inferred results with each trained model

are available at [onedrive](https://liveuniba-my.sharepoint.com/:f:/g/personal/tuch1_uniba_sk/IgAxWRwEvroTTqAf-8p0UqblAYIXUwGzlCyU-3BmcvwE5F8?e=otBeeM).


## how to reproduce?

1. install the required packages (see [requirements.txt](requirements.txt))
2. create the dataset by running the scripts in [01_dataset](01_dataset):

    1. [01_dataset/01_clean_file.py](01_dataset/01_clean_file.py) cleans the input text by removing blank lines, unwanted charactes, etc
    2. [01_dataset/02_preprocess_text.py](01_dataset/02_preprocess_text.py) divides the cleaned text into chunks of desired length
    3. [01_dataset/03_generate_dataset.py](01_dataset/03_generate_dataset.py) creates the dataset by rendering the text chunks into images of desired resolution and adding paper textures, stains, blur and other degradations. for each text chunk, a clean image, a degraded image, and ground truth text are created
    4. [01_dataset/04_split_dataset.py](01_dataset/04_split_dataset.py) splits the created dataset into train, val and test sets

3. train the networks in [02_networks](02_networks) by running the script:

    1. [02_networks/01_train.py](02_networks/01_train.py) trains the desired image enhancement model (drunet, dncnn, pix2pix) on the created dataset

4. evaluate the trained models by running the scripts in [03_evaluation](03_evaluation):

    1. [03_evaluation/01_inference.py](03_evaluation/01_inference.py) runs inference on the test set using the trained models
    2. [03_evaluation/02_ocr.py](03_evaluation/02_ocr.py) runs OCR on the inferred images and saves the extracted texts
    3. [03_evaluation/03_analysis.py](03_evaluation/03_analysis.py) computes character error rate (CER), and confusion matrices by comparing the extracted texts with the ground truth texts
    4. [03_evaluation/04_visualize_confusion_matrices.py](03_evaluation/04_visualize_confusion_matrices.py) visualizes the confusion matrices
    5. [03_evaluation/05_safety_evaluation.py](03_evaluation/05_safety_evaluation.py) evaluates the safety of the trained models based on the insertion, deletion, and substitution errors obtained from the confusion matrices
    6. [03_evaluation/06_iqa.py](03_evaluation/06_iqa.py) computes image quality assessment metrics (PSNR, SSIM, MSE) on clean images and the inferred images
