
#? Preprocessing functions for the dataset START

import pandas as pd
def Outlier_Removal(df_train, OD_majority, OD_minority): 
    cont_cols = ['Age', 'UAlb', 'Ucr', 'UACR', 'TC', 'TG', 'TCTG', 
                 'LDLC', 'HDLC', 'Scr', 'BUN', 'FPG', 'HbA1c', 'Height', 'Weight', 'BMI', 'Duration']
    # Use the original encoded single column name here
    cat_cols = ['Gender', 'Community'] 
    y_col = 'DR'

    print("Original class distribution:",df[y_col].value_counts())
    assert y_col in df_train.columns, f"'{y_col}' column is missing in the DataFrame."
    
    #* OUTLIER DETECTION START
    available_cont_cols = [col for col in cont_cols if col in df.columns]
    df_majority = df[df[y_col] == 0].copy()
    df_minority = df[df[y_col] == 1].copy()
    if OD_majority is not None:
        outliers_majority = OD_majority.fit_predict(df_majority[available_cont_cols])
        df_majority = df_majority[outliers_majority == 1]
        print(f"After OD, majority: {len(df_majority)}")
    if OD_minority is not None:
        outliers_minority = OD_minority.fit_predict(df_minority[available_cont_cols])
        df_minority = df_minority[outliers_minority == 1]
        print(f"After OD, minority: {len(df_minority)}")
    df_after_OD = pd.concat([df_majority, df_minority], ignore_index=True)
    #* OUTLIER DETECTION END - df_after_OD is the new df
    return df_after_OD

import sdv
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.sampling import Condition

def Synthetic_Data_Generator(df_train, synthesizer = "TVAE", conditions = None, epochs = 200, batch_size = 512, n_synthetic_data = 1000): 
    """Conditions: "balanced" or None"""
    metadata = Metadata.detect_from_dataframe(data=df_train)
    metadata.validate()
    metadata.visualize()
    #* Synthetic Data generation conditions
    condition_list = []
    if conditions == "balanced":
        Balanced = Condition(
                            num_rows=df_train[df_train['DR'] == 0].shape[0],
                            column_values={'DR': '1'}
                            )
        print("Balancing condition applied: adding DR=1 samples only")
        condition_list.append(Balanced)
    elif conditions == None:
        synthetic_data_count = Condition(
                            num_rows=n_synthetic_data,
                            )
        print("Generating {n_synthetic_data} samples without conditions")
        condition_list.append(synthetic_data_count)
        
    #* Synthesizer setup
    if synthesizer == "CTGAN":
        synthesizer = CTGANSynthesizer(
                                metadata=metadata, 
                                enforce_min_max_values=True, 
                                enforce_rounding=True, 
                                epochs = epochs,
                                verbose=True, 
                                cuda=True,
                                batch_size=batch_size
                                )   
    elif synthesizer == "TVAE":
        synthesizer = TVAESynthesizer(
                                metadata=metadata, 
                                enforce_min_max_values=True, 
                                enforce_rounding=True, 
                                epochs = epochs,
                                verbose=True, 
                                cuda=True,
                                batch_size=batch_size,
                                )
    else:
        return df_train
    synthetic_data = synthesizer.sample_from_conditions(
        conditions=condition_list,
        output_file_path='./synthetic_dataset/synthetic_data.csv'
    )
    df_train = pd.concat(synthetic_data, df_train, ignore_index=True)
    return df_train

from imblearn.over_sampling import SMOTENC
def apply_smotenc_oversampling(df_train):
    
    cont_cols = ['Age', 'UAlb', 'Ucr', 'UACR', 'TC', 'TG', 'TCTG', 
                 'LDLC', 'HDLC', 'Scr', 'BUN', 'FPG', 'HbA1c', 'Height', 'Weight', 'BMI', 'Duration']
    # Use the original encoded single column name here
    cat_cols = ['Gender', 'Community'] 
    y_col = 'DR'

    print("\nApplying SMOTENC oversampling...")

    # Split features and label
    X = df_train.drop(columns=[y_col])
    y = df_train[y_col]

    # Find indices of categorical features
    cat_indices = [X.columns.get_loc(col) for col in cat_cols if col in X.columns]

    # Ensure 'Community' is integer type if present
    if 'Community' in X.columns:
        X['Community'] = X['Community'].astype(int)

    oversampler = SMOTENC(categorical_features=cat_indices, random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    print(pd.DataFrame(X_resampled, columns=X.columns).describe())
    print("\nFinal class distribution:", pd.Series(y_resampled).value_counts())

    # Recombine into a single DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[y_col] = y_resampled

    return df_resampled

def apply_one_hot_encoding(df_train, df_test):
    community_mapping = {
    0: 'Community_baihe', 1: 'Community_chonggu', 2: 'Community_huaxin', 3: 'Community_jinze', 4: 'Community_liantang', 5: 'Community_xianghuaqiao', 6: 'Community_xujin', 7: 'Community_yingpu', 8: 'Community_zhaoxian', 9: 'Community_zhujiajiao'
    }
    
    # Map integer community labels to names
    for df in [df_train, df_test]:
        if 'Community' in df.columns:
            df['Community'] = df['Community'].astype(int).map(community_mapping)

    # One-hot encode the 'Community' column
    for df in [df_train, df_test]:
        if 'Community' in df.columns:
            df = pd.get_dummies(df, columns=['Community'], prefix='Community', prefix_sep='_', drop_first=False, dtype=int)

    # Align test set to training columns
    final_train_cols = df_train.columns

    df_test = df_test.reindex(columns=final_train_cols, fill_value=0)

    return df_train, df_test

def get_bmi(df, df_test):
    # Calculate BMI for both training and test sets
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df_test['BMI'] = df_test['Weight'] / ((df_test['Height'] / 100) ** 2)
    return df, df_test

from sklearn.model_selection import StratifiedKFold
def FOLDS_GENERATOR(dataset, n_splits=5, random_state=None, 
                    OD_majority=None, OD_minority=None,
                    oversampler_first = True, oversampler=None, 
                    synthesizer = "TVAE", epochs = 200, n_synthetic_data=None, 
                    scaler=None):
    
    cont_cols = ['Age', 'UAlb', 'Ucr', 'UACR', 'TC', 'TG', 'TCTG', 
                 'LDLC', 'HDLC', 'Scr', 'BUN', 'FPG', 'HbA1c', 'Height', 'Weight', 'BMI', 'Duration']
    # Use the original encoded single column name here
    cat_cols = ['Gender', 'Community'] 
    y_col = 'DR'
    
    
    kF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kFolds_list = []

    # Convert column names to strings to ensure compatibility
    df = dataset.copy()
    X = df.drop(columns=["DR"])
    Y = pd.DataFrame(df["DR"])

    for fold, (train_idx, test_idx) in enumerate(kF.split(X, Y)):
        # Split the data into training and testing sets for this fold
        train = pd.concat([X.iloc[train_idx], Y.iloc[train_idx]], axis=1)
        test = pd.concat([X.iloc[test_idx], Y.iloc[test_idx]], axis=1)
        
        #* OUTLIER DETECTION
        X_train_processed = Outlier_Removal(train, 
                                            OD_majority=OD_majority,
                                            OD_minority=OD_minority,
                                            )
        
        #* OVERSAMPLING & SYNTHETIC DATA GENERATION
        print("Before oversampling & synthetic data:", X_train_processed[["DR"]].value_counts())
        if oversampler_first: 
            X_train_processed = apply_smotenc_oversampling(X_train_processed)
            X_train_processed = Synthetic_Data_Generator(X_train_processed, synthesizer=synthesizer, conditions="None", epochs=epochs, batch_size=512, n_synthetic_data=n_synthetic_data)
        else:
            X_train_processed = Synthetic_Data_Generator(X_train_processed, synthesizer=synthesizer, conditions="None", epochs=epochs, batch_size=512, n_synthetic_data=None)
            X_train_processed = apply_smotenc_oversampling(X_train_processed)
        print("After oversampling & synthetic data:", X_train_processed[["DR"]].value_counts())
        
        #* Calculate BMI & ENCODING
        X_train_processed, test = get_bmi(X_train_processed, test)
        X_train_processed, test = apply_one_hot_encoding(X_train_processed, test)
        #* Scaler
        X_train_processed[cont_cols] = scaler.fit_transform(X_train_processed[cont_cols])
        test[cont_cols] = scaler.transform(test[cont_cols])
        # Append processed data (excluding the target column 'DR')
        kFolds_list.append((X_train_processed.drop(columns=["DR"]),
                            test.drop(columns=["DR"]),
                            X_train_processed[["DR"]],
                            test[["DR"]]))

        print(f"Fold: {fold+1}, Train: {X_train_processed.shape}, Test: {test.shape}")
    
    return kFolds_list

#? Preprocessing functions for the dataset END