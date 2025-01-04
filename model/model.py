import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Function to train the machine learning model
def create_model(data):
    # Separate the features (X) and target variable (y)
    X = data.drop(['id', 'diagnosis'], axis=1)
    y = data['diagnosis']

    # Handle missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Standardize the features for better performance of models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=0.95)  # Retain enough components to explain 95% variance
    X_pca = pca.fit_transform(X_scaled)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Initialize individual models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)  # Random Forest
    ann = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)  # Artificial Neural Network
    ect = ExtraTreesClassifier(n_estimators=100, random_state=42)  # Extra Trees Classifier

    # Create an ensemble model using a VotingClassifier (soft voting)
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf), ('ann', ann), ('ect', ect)],
        voting='soft'  # Use probabilities for voting
    )

    # Train the ensemble model on the training data
    ensemble_model.fit(X_train, y_train)

    # Evaluate the model on the test data
    y_pred = ensemble_model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))

    return ensemble_model, scaler, imputer, pca

# Function to load and preprocess the data
def get_clean_data():
    # Load the dataset
    data = pd.read_csv("data/data.csv")
    # Drop unnecessary columns
    data = data.drop(['Unnamed: 32'], axis=1, errors='ignore')
    # Map the target variable: Malignant ('M') as 1, Benign ('B') as 0
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def main():
    # Load the cleaned dataset
    data = get_clean_data()

    # Train the model and get the trained scaler, imputer, and PCA
    ensemble_model, scaler, imputer, pca = create_model(data)

    # Create the model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)

    # Save the trained model using joblib
    joblib.dump(ensemble_model, 'model/ensemble_model.joblib')

    # Save the scaler using joblib
    joblib.dump(scaler, 'model/scaler.joblib')

    # Save the imputer using joblib
    joblib.dump(imputer, 'model/imputer.joblib')

    # Save the PCA using joblib
    joblib.dump(pca, 'model/pca.joblib')

if __name__ == '__main__':
    main()
