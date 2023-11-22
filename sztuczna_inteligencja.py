import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate_model(data_path='Heart Attack.csv', target_column='class', test_size=0.2, random_state=0):
    # Wczytujemy dane z pliku CSV
    df = pd.read_csv(data_path)

    # Dzielimy dane na zestawy treningowe i testowe
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Tworzymy model RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Dopasowujemy model do danych treningowych
    model.fit(X_train, y_train)

    # Przewidujemy klasy dla danych testowych
    y_pred = model.predict(X_test)

    # Sprawdzamy dok³adnoœæ modelu
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Dokladnosc modelu: {accuracy:.4f}')

    # Wyœwietlamy raport klasyfikacji
    print('Raport klasyfikacji:')
    print(classification_report(y_test, y_pred))
    return accuracy, model  # Zwracamy równie¿ model

# U¿ycie funkcji z domyœlnymi parametrami
licznik = 30
best_accuracy = 0
best_model = None

while licznik <= 40:
    print(licznik)
    current_accuracy, current_model = train_and_evaluate_model(random_state=licznik)
    
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_model = current_model

    licznik += 1

# Teraz mo¿esz u¿yæ najlepszego modelu poza pêtl¹
# Przyk³ad u¿ycia wytrenowanego modelu do przewidzenia nowych danych
new_data = {
    'age': 45,
    'gender': 1,
    'impluse': 2,
    'pressurehight': 140,
    'pressurelow': 140,
    'glucose': 260,
    'kcm': 0,
    'troponin': 1
}

# Tworzymy DataFrame z nowymi danymi
new_data_df = pd.DataFrame([new_data])

# Przewidujemy wynik za pomoc¹ najlepszego modelu
predicted_result = best_model.predict(new_data_df)

print(f'Przewidywany wynik dla nowych danych: {predicted_result}')
