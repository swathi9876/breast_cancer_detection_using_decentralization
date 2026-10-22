

from common_modules import*
import pandas as pd
# Load and preprocess the dataset
def preprocess_data(data_path):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Drop any rows with missing values
    df = df.dropna()

    # Label encoding for the "diagnosis" column
    label_encoder = LabelEncoder()
    df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

    # Dropping ID column
    df = df.drop(df.columns[0], axis=1)

    # Extract features and labels
    X = df.iloc[:, 1:].values  # Features
    y = df.iloc[:, 0].values   # Labels (0 or 1)

    # Scale features using StandardScaler to ensure stable training and high accuracy
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
