# Near Duplicate Video Detection

This project detects near-duplicate videos using deep metric learning and a Qdrant vector database. It generates embeddings with AlexNet, the early fusion model is trained on the VCDB dataset.

## Repository Structure
    Notebooks:
        - dataset.ipynb: Dataset preprocessing.
        - embeddings.ipynb: Feature extraction and video embedding generation.
        - pipeline.ipynb: Model training, evaluation, and duplicate detection.

## Setup

Clone the repo.
Install dependencies:

    pip install -r requirements.txt

## Train
If you want to train a model yourself, download the core VCDB dataset from the [VCDB website](https://fvl.fudan.edu.cn/dataset/vcdb/list.htm). Then, go through the embeddings.ipynb and pipeline.ipynb notebooks.

## Usage
