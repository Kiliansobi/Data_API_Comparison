import openml
import pandas as pd

def create_metadata(ids_array):
    res = []

    for i in range(0, len(ids_array) - 1):
            dataset = openml.datasets.get_dataset(ids_array[i], download_data = False)
            meta_data = {
                'file_name': dataset.name,
                'num_instances': dataset.qualities.get('NumberOfInstances'),
                'num_features': dataset.qualities.get('NumberOfFeatures'), 
                'num_missing_values': dataset.qualities.get('NumberOfMissingValues'),
                'num_instances_with_missing': dataset.qualities.get('NumberOfInstancesWithMissingValues'),
                'num_numeric_features': dataset.qualities.get('NumberOfNumericFeatures'),
                'num_categorical_features': dataset.qualities.get('NumberOfSymbolicFeatures')
            }
            
            res.append(meta_data)
            
    return pd.DataFrame(res)