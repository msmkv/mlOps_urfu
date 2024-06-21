# DON'T FORGET INSTALL DEPENDENCIES
# pip install catboost PANDAS
import pandas as pd
import os
from catboost.datasets import titanic


root_path = ".."
dataset_path = "titanic-dataset"
train_path, test_path = "train.csv", "test.csv"

# CREATE DATASET DIR IF DOES NOT EXIST
if not os.path.exists(os.path.join(root_path, dataset_path)):
    os.mkdir(os.path.join(root_path, dataset_path))

full_train_path = os.path.join(root_path, dataset_path, train_path)
full_test_path = os.path.join(root_path, dataset_path, test_path)

# GET DATA
def create_and_save_dataset(file_path, number_of_dataset):
    if not os.path.exists(file_path):
        titanic_ds = titanic()[number_of_dataset]
        df = pd.DataFrame(titanic_ds)
        df.to_csv(file_path)
    else:
        print(f"File already exists - {file_path}") 


# CHECK DATA
def check_file(file_path):
    if os.path.exists(file_path):
        print(f"File exists - {file_path}")
    else:
        prinit(f"Fail downloading - {file_path}")


create_and_save_dataset(full_train_path, 0)
create_and_save_dataset(full_test_path, 1)
check_file(full_train_path)
check_file(full_test_path)
