# DON'T FORGET INSTALL DEPENDENCIES
# pip install numpy pandas
import numpy as np
import pandas as pd


train_path, test_path = "../titanic-dataset/train.csv", "../titanic-dataset/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


# FILL NULL ROWS FOR AGE COLUMN
def fill_mean_value_for_age_column(df, df_path):
    if df['Age'].isna().sum() > 0:
        df['Age'] = df['Age'].fillna(int(np.mean(df['Age'])))
        print(f"Filled age column: {df_path}")
    else:
        print(f"Nothing for filling on age column: {df_path}")


# CHECK COLUMN AND SAVE DF
def check_column_on_empty_rows_and_save(df, df_path):
    if df['Age'].isna().sum() == 0:
        print(f"Age column is ready for use: {df_path}")
        df.to_csv(df_path)
    else:
        print(f"Age column is not ready for use: {df_path}")


fill_mean_value_for_age_column(train_df, train_path)
fill_mean_value_for_age_column(test_df, test_path)

check_column_on_empty_rows_and_save(train_df, train_path)
check_column_on_empty_rows_and_save(test_df, test_path)

