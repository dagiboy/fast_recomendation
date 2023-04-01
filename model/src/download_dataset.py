import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files('otto/recsys-dataset', path='data', unzip=True, quiet=False)