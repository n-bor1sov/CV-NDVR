# Near Duplicate Video Detection

This project detects near-duplicate videos using deep metric learning and a Qdrant vector database. It generates 
embeddings with AlexNet, the early fusion model is trained on the VCDB dataset.

## Repository Structure

    Notebooks:
        - dataset.ipynb: Dataset preprocessing.
        - embeddings.ipynb: Feature extraction and video embedding generation.
        - model.ipynb: Model training, evaluation, and duplicate detection.
    Api:
        -model_77.pt: Model weights.
        -api.py: Flask api.
    Front:
        -app.py: Streamlit application.

    -pipeline.ipynb: Overlook of the code output as a sample running.

## Setup

Clone the repo.
Install dependencies:

    pip install -r requirements.txt

## Train

If you want to train a model yourself, download the core VCDB dataset from the 
[VCDB website](https://fvl.fudan.edu.cn/dataset/vcdb/list.htm) and unzip it to the project directory.
Then, go through the embeddings.ipynb and pipeline.ipynb notebooks.



## Usage
1) Run the local qdrant database
```
docker pull qdrant/qdrant
sudo docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```
2) Run flask you can use command `python api/api.py` from the root of repository

3) Run the streamlit app you can use command `streamlit run api/app.py`

After sending the video to the api or uploading it to the streamlit app, its vector representation will be saved in the
database and used for duplicate detection. 
If the duplicate is found, the new video would not be added to the database.