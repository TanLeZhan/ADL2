import pandas as pd
from sklearn.preprocessing import StandardScaler

def gender_to_ones_and_zeros(df):
    if df["Gender"].isin([0, 1]).all():
        print("gender column already in 0 and 1")
        return df
    df[["Gender"]] = df[["Gender"]].replace({1: 0, 2: 1})
    return df

from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(input_file="raw_data.csv", 
                              output_file="processed_data_OHE.csv", 
                              output_file2="processed_data_encoded.csv"): 
    """input_file: str, path to the raw data file
    output_file: str, path to save the processed data file (one-hot)
    output_file2: str, path to save the label-encoded version
    """
    df = pd.read_csv(input_file)
    
    df.rename(columns={
        'age': 'Age', 'gender': 'Gender', 'community': 'Community', 'U-Alb': 'UAlb', 
        'LDL-C': 'LDLC', 'HDL-C': 'HDLC', 'ACR': 'UACR'
    }, inplace=True)

    continuous_features = ['Age', 'UAlb', 'Ucr', 'UACR', 'TC', 'TG', 'TCTG', 'LDLC', 'HDLC', 
                           'Scr', 'BUN', 'FPG', 'HbA1c', 'Height', 'Weight', 'BMI', 'Duration']
    
    for col in continuous_features:
        df[col] = df[col].astype(str).str.replace(',', '')  
        df[col] = pd.to_numeric(df[col], errors='coerce') 
    
    df[continuous_features] = df[continuous_features].fillna(df[continuous_features].median())

    df = df[["Age", "Gender", "Community", "UAlb", "Ucr", "UACR", "TC", "TG", "TCTG", "LDLC", 
             "HDLC", "Scr", "BUN", "FPG", "HbA1c", "Height", "Weight", "BMI", "Duration", "DR"]]
    
    # One-hot version
    df_onehot = pd.get_dummies(df, columns=['Community'], dtype=float)
    df_onehot = gender_to_ones_and_zeros(df_onehot)
    df_onehot.to_csv(output_file, index=False)
    print(f"Processed (one-hot) data saved to {output_file}")

    # Label-encoded version
    df_encoded = df.copy()
    le = LabelEncoder()
    df_encoded['Community'] = le.fit_transform(df_encoded['Community'])
    df_encoded = gender_to_ones_and_zeros(df_encoded)
    df_encoded.to_csv(output_file2, index=False)
    print(f"Processed (label-encoded) data saved to {output_file2}")

def Class_Split(dataset):
    """Takes in a dataset and splits it into positive and negative samples,
    returns positive, negative df"""
    Positive = dataset[dataset['DR'] == 1].reset_index()
    Negative = dataset[dataset['DR'] == 0].reset_index()
    return Positive, Negative




if __name__ == "__main__":
    load_and_preprocess_data()
    
