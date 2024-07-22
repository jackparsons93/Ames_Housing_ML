# data_processing_module.py

# 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 2: Define the load_data function
def load_data(file_path):
    """
    Load the dataset from the given file path.
    """
    data = pd.read_csv(file_path)
    return data

# 3: Define the preprocess_data function
def preprocess_data(data, target_column, drop_columns):
    """
    Preprocess the data by separating features and target, 
    identifying categorical and numerical columns, 
    and applying transformations.
    """
    # Separate features and target
    X = data.drop([target_column], axis=1)
    y = data[target_column]
    
    # Remove unnecessary columns
    X = X.loc[:, ~X.columns.isin(drop_columns)]
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns
    
    # Create preprocessing pipelines for both numeric and categorical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Preprocess the data
    X_preprocessed = preprocessor.fit_transform(X)
    
    return X_preprocessed, y, preprocessor

# 4: Define the split_data function
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage:
    file_path = 'path_to_your_data.csv'
    target_column = 'target'
    drop_columns = ['unnecessary_column1', 'unnecessary_column2']
    
    # Load data
    data = load_data(file_path)
    
    # Preprocess data
    X_preprocessed, y, preprocessor = preprocess_data(data, target_column, drop_columns)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X_preprocessed, y)
    
    print("Data loaded, preprocessed, and split into training and testing sets.")