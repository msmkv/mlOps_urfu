import os


train_path, test_path = "../titanic-dataset/train.csv", "../titanic-dataset/test.csv"


# DELETE DATA
def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print(f"Cannot find file for removing - {file_path}")


# CHECK DATA
def check_file(file_path):
    if not os.path.exists(file_path):
        print(f"Success removing - {file_path}")
    else:
        print(f"Fail removing - {file_path}")



remove_file(train_path)
check_file(train_path)

remove_file(test_path)
check_file(test_path)

