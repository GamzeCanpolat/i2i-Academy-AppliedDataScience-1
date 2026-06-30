import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# =====================================================
# 1. Veri Setini Yükleme
# =====================================================

cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target

# =====================================================
# 2. Veri Setini İnceleme
# =====================================================

print("=" * 60)
print("İLK 5 SATIR")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("VERİ BOYUTU")
print("=" * 60)
print(df.shape)

print("\n" + "=" * 60)
print("SÜTUN İSİMLERİ")
print("=" * 60)
print(df.columns)

print("\n" + "=" * 60)
print("VERİ BİLGİLERİ")
print("=" * 60)
df.info()

print("\n" + "=" * 60)
print("İSTATİSTİKSEL ÖZET")
print("=" * 60)
print(df.describe())

# =====================================================
# 3. Eksik Veri Kontrolü
# =====================================================

print("\n" + "=" * 60)
print("EKSİK VERİ KONTROLÜ")
print("=" * 60)
print(df.isnull().sum())

# =====================================================
# 4. Feature ve Target Ayırma
# =====================================================

X = df.drop("target", axis=1)
y = df["target"]

print("\n" + "=" * 60)
print("VERİ BOYUTLARI")
print("=" * 60)
print("X Boyutu :", X.shape)
print("y Boyutu :", y.shape)

# =====================================================
# 5. Train - Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 60)
print("TRAIN - TEST SPLIT")
print("=" * 60)
print("Eğitim Verisi :", X_train.shape)
print("Test Verisi   :", X_test.shape)

# =====================================================
# 6. Feature Scaling
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling başarıyla uygulandı.")

# =====================================================
# 7. Logistic Regression
# =====================================================

log_model = LogisticRegression(random_state=42)

log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

accuracy_log = accuracy_score(y_test, y_pred_log)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION SONUÇLARI")
print("=" * 60)

print(f"Accuracy Score : %{accuracy_log * 100:.2f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

# =====================================================
# 8. Random Forest
# =====================================================

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("\n" + "=" * 60)
print("RANDOM FOREST SONUÇLARI")
print("=" * 60)

print(f"Accuracy Score : %{accuracy_rf * 100:.2f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# =====================================================
# 9. Modellerin Karşılaştırılması
# =====================================================

print("\n" + "=" * 60)
print("MODEL KARŞILAŞTIRMASI")
print("=" * 60)

print(f"Logistic Regression Accuracy : %{accuracy_log * 100:.2f}")
print(f"Random Forest Accuracy       : %{accuracy_rf * 100:.2f}")

if accuracy_log > accuracy_rf:
    best_model = "Logistic Regression"
elif accuracy_rf > accuracy_log:
    best_model = "Random Forest"
else:
    best_model = "Her iki model de aynı başarıyı gösterdi."

print(f"\nEn başarılı model: {best_model}")

print("\nProgram başarıyla tamamlandı.")