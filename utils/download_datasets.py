from pathlib import Path

import opendatasets as od

dataset_url = 'https://www.kaggle.com/datasets/nih-chest-xrays/sample'

# Look into the data directory
datasets = 'datasets/sample'
dataset_path = Path(datasets)

dataset_path.mkdir(parents=True, exist_ok=True)
if not dataset_path.is_dir():
    od.download(dataset_url)
