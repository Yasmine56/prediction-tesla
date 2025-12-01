import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf


from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
 accuracy_score, classification_report, confusion_matrix,
 roc_curve, roc_auc_score, precision_recall_curve
)

# === 1. Récupération des données ===
ticker = 'TSLA'
df = yf.Ticker(ticker).history(period='5y')

# === 2. Feature engineering simplifié (utilisé par les deux modèles) ===
df['RSI'] = df['Close'].diff().apply(lambda x: max(x, 0)).rolling(14).mean() / \
 df['Close'].diff().apply(lambda x: abs(min(x, 0))).rolling(14).mean()
df['RSI'] = 100 - (100 / (1 + df['RSI']))
df['RSI_Mean'] = df['RSI'].rolling(window=14).mean()
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['Volatility'] = df['Close'].rolling(window=10).std()

# Variable cible
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

# === 3. Préparation des données avec fenêtre glissante ===
def create_features(data, window_size):
 X, y = [], []
 features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'RSI_Mean', 'SMA_10', 'SMA_50', 'Volatility']
 for i in range(len(data) - window_size):
 X.append(data[features].iloc[i:i+window_size].values.flatten())
 y.append(data['Target'].iloc[i+window_size])
 return np.array(X), np.array(y)

window_size = 30
X, y = create_features(df, window_size)

# === 4. Split + Standardisation ===
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 5. Entraînement des deux modèles ===
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
lr_model = LogisticRegression(max_iter=1000)

dt_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# === 6. Prédictions ===
models = {
 "Decision Tree": (dt_model, dt_model.predict(X_test), dt_model.predict_proba(X_test)[:, 1]),
 "Logistic Regression": (lr_model, lr_model.predict(X_test), lr_model.predict_proba(X_test)[:, 1])
}

from sklearn.metrics import (
 accuracy_score, classification_report, confusion_matrix,
 roc_curve, roc_auc_score, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# === 7. Évaluations communes ===

# Courbe ROC comparée
plt.figure(figsize=(8, 6))
for name, (model, y_pred, y_proba) in models.items():
 fpr, tpr, _ = roc_curve(y_test, y_proba)
 auc_score = roc_auc_score(y_test, y_proba)
 plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Hasard')
plt.title('Courbe ROC comparée')
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Courbe Precision-Recall comparée
plt.figure(figsize=(8, 6))
for name, (model, y_pred, y_proba) in models.items():
 precision, recall, _ = precision_recall_curve(y_test, y_proba)
 plt.plot(recall, precision, label=name)
plt.title('Courbe Precision-Recall comparée')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Matrices de confusion et rapport de classification
for name, (model, y_pred, y_proba) in models.items():
 print(f"\n=== {name} ===")
 print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
 print("Classification Report:\n", classification_report(y_test, y_pred))

 cm = confusion_matrix(y_test, y_pred)
 plt.figure(figsize=(6, 4))
 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
 plt.title(f'Matrice de confusion - {name}')
 plt.xlabel('Prédit')
 plt.ylabel('Réel')
 plt.tight_layout()
 plt.show()
