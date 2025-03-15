import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
import os

def search_datasets_tags(tags_id):
    api = KaggleApi()
    api.authenticate()

    page = 1
    identifiers = []

    while page < 50:
        datasets = api.dataset_list(tag_ids = tags_id,
                                    page = page,
                                    max_size = 1048576*100,
                                    file_type = "csv")
        if not datasets:
            break

        identifiers.extend([f"{dataset.ref}" for dataset in datasets])
        page += 1

    return identifiers

def search_datasets(search_term):
    api = KaggleApi()
    api.authenticate()

    page = 1
    identifiers = []

    while page <= 50:
        datasets = api.dataset_list(search = search_term, page = page, max_size = 1048576*100)
        if not datasets:
            break

        identifiers.extend([f"{dataset.ref}" for dataset in datasets])
        page += 1

    return identifiers

def download_file(id_sample, download_path):
    api = KaggleApi()
    api.authenticate()

    # download
    api.dataset_download_files(id_sample,
                              path = download_path,
                              unzip = True)
    
def analyse_file(file_path):
    df = pd.read_csv(file_path, encoding_errors = 'ignore', low_memory = False, on_bad_lines='skip')

    # analyse
    metadata = {
       'file_name': os.path.basename(file_path),
        'num_instances': df.shape[0],
        'num_features': df.shape[1],
        'num_missing_values': df.isnull().sum().sum(),
        'num_instances_with_missing': df.isnull().any(axis=1).sum(),
        'num_numeric_features': df.select_dtypes(include=np.number).shape[1],
        'num_categorical_features': df.select_dtypes(exclude=np.number).shape[1] 
    }

    return metadata

def create_metadata(ids_array, download_path):

    # create download directory
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    # download files
    for i in range(0, len(ids_array)-1):
        download_file(ids_array[i], download_path)

    # analyse files
    res = []

    for file in os.listdir(download_path):
        if file.endswith('.csv'):
            csv_file = os.path.join(download_path, file)
            try:
                meta_data = analyse_file(csv_file)
                res.append(meta_data)
            except Exception as e:
                print(f"file: {file}, error: {e}")

    return pd.DataFrame(res)
