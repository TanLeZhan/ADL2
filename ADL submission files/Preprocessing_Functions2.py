import pandas as pd
from sklearn.base import BaseEstimator
import numpy as np
import sdv
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.sampling import Condition
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.evaluation.single_table import get_column_plot
import os
def apply_one_hot_encoding(df_train, df_test):
    community_mapping = {
        0: 'Community_baihe', 1: 'Community_chonggu', 2: 'Community_huaxin', 3: 'Community_jinze',
        4: 'Community_liantang', 5: 'Community_xianghuaqiao', 6: 'Community_xujin', 7: 'Community_yingpu',
        8: 'Community_zhaoxian', 9: 'Community_zhujiajiao'
    }

    # Map integer community labels to names
    for df in [df_train, df_test]:
        if 'Community' in df.columns:
            df['Community'] = df['Community'].astype(int).map(community_mapping)

    # One-hot encode the 'Community' column — assign back properly!
    if 'Community' in df_train.columns:
        df_train = pd.get_dummies(df_train, columns=['Community'], prefix='Community', prefix_sep='_', drop_first=False, dtype=int)

    if 'Community' in df_test.columns:
        df_test = pd.get_dummies(df_test, columns=['Community'], prefix='Community', prefix_sep='_', drop_first=False, dtype=int)

    # Align test set to training columns
    final_train_cols = df_train.columns

    df_test = df_test.reindex(columns=final_train_cols, fill_value=0)

    return df_train, df_test

def Outlier_Removal(df_train, OD_majority, OD_minority): 
    cont_cols = ['Age', 'UAlb', 'Ucr', 'UACR', 'TC', 'TG', 'TCTG', 
                 'LDLC', 'HDLC', 'Scr', 'BUN', 'FPG', 'HbA1c', 'Height', 'Weight', 'BMI', 'Duration']
    # Use the original encoded single column name here
    cat_cols = ['Gender', 'Community'] 
    y_col = 'DR'

    print("Original class distribution:",df_train[y_col].value_counts())
    assert y_col in df_train.columns, f"'{y_col}' column is missing in the DataFrame."
    
    #* OUTLIER DETECTION START
    available_cont_cols = [col for col in cont_cols if col in df_train.columns]
    df_majority = df_train[df_train[y_col] == 0].copy()
    df_minority = df_train[df_train[y_col] == 1].copy()
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

class IQRDetector(BaseEstimator):
    def __init__(self, factor=1.5):
        """
        Factor    Effect                          Use Case
        -------   ------------------------------- -------------------------------
        1.0       Very sensitive                   Noisy data, you want to clean aggressively
        1.5       (default) Balanced               Standard statistical practice
        2.0-3.0   Less sensitive                   Conservative — only extreme points flagged
        """
        self.factor = factor
        self.columns_ = None
        self.bounds_ = {}

    def fit(self, X, y=None):
        self.columns_ = X.columns
        for col in self.columns_:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.factor * IQR
            upper = Q3 + self.factor * IQR
            self.bounds_[col] = (lower, upper)
        return self

    def predict(self, X):
        is_outlier = pd.Series([False] * X.shape[0], index=X.index)
        for col in self.columns_:
            lower, upper = self.bounds_[col]
            is_outlier |= (X[col] < lower) | (X[col] > upper)
        return np.where(is_outlier, -1, 1)  # -1 = outlier, 1 = inlier

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)
    
def Synthetic_Data_Generator(df_train, fold, synthesizer = "TVAE", epochs = 200, batch_size = 128, n_synthetic_data = 1000): 
    """Conditions: "balanced" or None"""
    # df_train = df_train.drop(columns=["BMI", "TCTG"])
    metadata = Metadata.detect_from_dataframe(data=df_train)
    metadata.validate()
    
    #* Synthetic Data generation conditions
    condition_list = []
    #* Synthesizer setup
    if synthesizer == "CTGAN":
        filepath = f"{synthesizer}_{epochs}.pkl"
        synthesizer = CTGANSynthesizer(
                                metadata=metadata, 
                                enforce_min_max_values=True, 
                                enforce_rounding=True, 
                                epochs = epochs,
                                verbose=True, 
                                cuda=True,
                                batch_size=300 # need to be divisible by 10 or pac size
                                )  
        # df_train = make_divisible(df_train, 10)
    elif synthesizer == "TVAE":
        filepath = f"{synthesizer}_{epochs}.pkl"
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
    print("Balancing condition applied")
    
    # Step 1: Fit the synthesizer
    synthesizer.fit(df_train)
    
    synthesizer.save(filepath)
    # Step 2: Get class counts
    # Step 1: Get class counts
    count_0 = df_train[df_train['DR'] == 0].shape[0]
    count_1 = df_train[df_train['DR'] == 1].shape[0]

    # Step 2: Balance to the max count
    balanced_per_class = max(count_0, count_1)
    print(balanced_per_class - count_0)
    print(balanced_per_class - count_1)
    cond_0 = Condition(column_values={'DR': 0}, num_rows=balanced_per_class - count_0)
    cond_1 = Condition(column_values={'DR': 1}, num_rows=balanced_per_class - count_1)

    balanced_data = synthesizer.sample_from_conditions([cond_0, cond_1])

    # Step 3: Add more *evenly* on top to hit n_synthetic_data
    # Note: You already have (balanced_per_class * 2) at this point

    current_total = balanced_per_class * 2
    remaining = n_synthetic_data - current_total

    # Split remaining evenly across classes
    extra_per_class = remaining // 2
    # print(extra_per_class)
    # print(remaining - extra_per_class)
    # (optional: +1 to one class if remaining is odd)
    cond_extra_0 = Condition(column_values={'DR': 0}, num_rows=extra_per_class)
    cond_extra_1 = Condition(column_values={'DR': 1}, num_rows=remaining - extra_per_class)

    extra_data = synthesizer.sample_from_conditions([cond_extra_0, cond_extra_1], max_tries_per_batch= 500, output_file_path="./DATA/synthetic_training_set/synthetic_data_conditions.csv")

    # Step 5: Combine all the synthetic garbage
    synthetic_data = pd.concat([balanced_data, extra_data], ignore_index=True)
    quality_report = evaluate_quality(df_train, synthetic_data, metadata)
    # synthetic_data = get_bmi_i(synthetic_data)
    # synthetic_data = get_TCTG_i(synthetic_data)
    
    # Ensure folder exists
    os.makedirs("./DATA/synthetic_training_set", exist_ok=True)
    # Save to specific file
    
    synthetic_data.to_csv(f"./DATA/synthetic_training_set/synthetic_data_{fold}_{epochs}_TVAE.csv", index=False)

    # synthetic_data.to_csv('./synthetic_dataset/synthetic_data2.csv', index=False)
    df_train = pd.concat([synthetic_data, df_train], ignore_index=True)
    return df_train

def Synthetic_Data_Generator2(df_train, fold, synthesizer="TVAE", epochs=200, batch_size=128, n_synthetic_data=1000):

    def safe_sample(synth, cond, target, max_retries=5):
        collected = []
        total = 0
        for _ in range(max_retries):
            batch = synth.sample_from_conditions([cond])
            collected.append(batch)
            total += batch.shape[0]
            if total >= target:
                break
        if total < target:
            print(f"⚠️ WARNING: Only got {total}/{target} for condition {cond.column_values}")
        return pd.concat(collected, ignore_index=True)

    metadata = Metadata.detect_from_dataframe(data=df_train)
    metadata.validate()

    if synthesizer == "CTGAN":
        filepath = f"{synthesizer}_{epochs}.pkl"
        synthesizer = CTGANSynthesizer(
            metadata=metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            epochs=epochs,
            verbose=True,
            cuda=True,
            batch_size=300
        )
    elif synthesizer == "TVAE":
        filepath = f"{synthesizer}_{epochs}.pkl"
        synthesizer = TVAESynthesizer(
            metadata=metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            epochs=epochs,
            verbose=True,
            cuda=True,
            batch_size=batch_size
        )
    else:
        return df_train

    print("Fitting synthesizer...")
    synthesizer.fit(df_train)
    synthesizer.save(filepath)

    # Calculate class proportions
    class_counts = df_train['DR'].value_counts(normalize=True)
    synthetic_data = []

    print("Generating synthetic samples per class based on distribution...")
    for cls, proportion in class_counts.items():
        n_samples = int(n_synthetic_data * proportion)
        cond = Condition(column_values={'DR': int(cls)}, num_rows=n_samples)
        sampled = safe_sample(synthesizer, cond, n_samples)
        synthetic_data.append(sampled)

    synthetic_data = pd.concat(synthetic_data, ignore_index=True)

    print("Final synthetic class distribution:")
    print(synthetic_data['DR'].value_counts())

    os.makedirs("./DATA/synthetic_training_set", exist_ok=True)
    synthetic_data.to_csv(f"./DATA/synthetic_training_set/synthetic_data_{fold}_{epochs}_{type(synthesizer).__name__}.csv", index=False)

    quality_report = evaluate_quality(df_train, synthetic_data, metadata)
    df_train = pd.concat([synthetic_data, df_train], ignore_index=True)
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

def get_bmi(df, df_test):
    # Calculate BMI for both training and test sets
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df_test['BMI'] = df_test['Weight'] / ((df_test['Height'] / 100) ** 2)
    return df, df_test

def get_TCTG(df, df_test):
    # Calculate TCTG for both training and test sets
    df['TCTG'] = df['TC'] / df['TG']
    df_test['TCTG'] = df_test['TC'] / df_test['TG']
    return df, df_test

from sklearn.model_selection import StratifiedKFold
def FOLDS_GENERATOR1(df:pd.DataFrame, n_splits=5, random_state=42, OD_majority=None, OD_minority=None,
                    synthesizer = "TVAE", epochs = 100, batch_size=512, n_synthetic_data=0,
                    scaler=None):
    """
    Generates n_splits folds for cross-validation.
    """
    cont_cols = ['Age', 'UAlb', 'Ucr', 'UACR', 'TC', 'TG', 'TCTG', 
                 'LDLC', 'HDLC', 'Scr', 'BUN', 'FPG', 'HbA1c', 'Height', 'Weight', 'BMI', 'Duration']
    cat_cols = ['Gender', 'Community'] 
    # Create a StratifiedKFold object
    kF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # Initialize a list to hold the folds
    df_copy = df.copy()
    X = df_copy.drop(columns=["DR"])
    Y = df[["DR"]]
    X = X.drop(columns=["BMI", "TCTG"])

    kFolds_list = []
    for fold, (train_idx, test_idx) in enumerate(kF.split(X, Y)):
        # Split the data into training and testing sets for this fold
        train = pd.concat([X.iloc[train_idx], Y.iloc[train_idx]], axis=1)
        test = pd.concat([X.iloc[test_idx], Y.iloc[test_idx]], axis=1)
        
        # # #* OUTLIER DETECTION
        # X_train_processed = Outlier_Removal(train, 
        #                                     OD_majority=OD_majority,
        #                                     OD_minority=OD_minority,
        #                                     )
        # X_train_processed = apply_smotenc_oversampling(X_train_processed)
        X_train_processed = train.copy()
        # #* OVERSAMPLING & SYNTHETIC DATA GENERATION
        print("Before oversampling & synthetic data:", X_train_processed[["DR"]].value_counts())
        X_train_processed = Synthetic_Data_Generator2(X_train_processed, fold, synthesizer=synthesizer, epochs=epochs, batch_size=256, n_synthetic_data=n_synthetic_data)
        # X_train_processed = Outlier_Removal(train, 
        #                                     OD_majority=OD_majority,
        #                                     OD_minority=OD_minority,
        #                                     )
        #
        X_train_processed = Outlier_Removal(X_train_processed, 
                                            OD_majority=OD_majority,
                                            OD_minority=OD_minority,
                                            )
        X_train_processed = apply_smotenc_oversampling(X_train_processed)
        
        print("After oversampling & synthetic data:", X_train_processed[["DR"]].value_counts())

        
        #* Calculate BMI, TCTG & ENCODING
        X_train_processed, test = get_bmi(X_train_processed, test)
        X_train_processed, test = get_TCTG(X_train_processed, test)
        X_train_processed, test = apply_one_hot_encoding(X_train_processed, test)
        #* Scaler
        X_train_processed[cont_cols] = scaler.fit_transform(X_train_processed[cont_cols])
        test[cont_cols] = scaler.transform(test[cont_cols])
        # Append processed data (excluding the target column 'DR')

        # Save to CSV with fold number
        X_train_processed.to_csv(f"./DATA/folds/train_fold_{fold}.csv", index=False)
        test.to_csv(f"./DATA/folds/test_fold_{fold}.csv", index=False)
        kFolds_list.append((
                            X_train_processed.drop(columns=['DR']),
                            test.drop(columns=['DR']),
                            X_train_processed['DR'].values.reshape(-1, 1),  # Ensures the target is 2D
                            test['DR'].values.reshape(-1, 1)  # Ensures the target is 2D
                        ))
        break
    print(f"Fold: {fold+1}, Train: {X_train_processed.drop(columns=['DR']).shape}, Test: {test.drop(columns=['DR']).shape}")
    
    return kFolds_list
