
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score


data = pd.read_csv('datos_lavado_activos.csv')



X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

model = IsolationForest(contamination=0.01) 


model.fit(X_train)


predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(predictions, [-1 if x == -1 else 1 for x in predictions]))
print("Classification Report:")
print(classification_report([-1 if x == -1 else 1 for x in predictions], [-1 if x == -1 else 1 for x in predictions]))
