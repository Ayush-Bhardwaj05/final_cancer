import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Define the individual models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    ann = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
    ect = ExtraTreesClassifier(n_estimators=100, random_state=42)

    # Create an ensemble of the models
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf), ('ann', ann), ('ect', ect)],
        voting='soft'
    )

    # Train the ensemble model
    ensemble_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = ensemble_model.predict(X_test)
    print('Accuracy of the ensemble model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    return ensemble_model, scaler


def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})  # Mapping 'M' -> 1, 'B' -> 0
    return data


def main():
    data = get_clean_data()

    ensemble_model, scaler = create_model(data)

    # Save the trained ensemble model and scaler
    with open('model/ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble_model, f)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    main()
