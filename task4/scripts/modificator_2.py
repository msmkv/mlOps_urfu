import pandas as pd


train_path, test_path = "../titanic-dataset/train.csv", "../titanic-dataset/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


# ONE-HOT-ENCODING FOR SEX COLUMN
def encode_sex_column(df, df_name):
    if 'Sex' in df.columns:
        return pd.get_dummies(df, columns=['Sex'], drop_first=True)
    else:
        print(f"Cannot find 'sex' column in dataset - {df_name}")
        return df


# CHECK SEX COLUMN AND SAVE
def check_column_and_save(df, df_path):
    if 'Sex' not in df.columns:
        print(f"Success encode for sex column - {df_path}")
        df.to_csv(df_path)
    else:
        print(f"Fail encode for sex column - {df_path}")


train_df = encode_sex_column(train_df, train_path)
check_column_and_save(train_df, train_path)

test_df = encode_sex_column(test_df, test_path)
check_column_and_save(test_df, test_path)
