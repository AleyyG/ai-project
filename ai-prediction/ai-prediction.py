# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
"""
# Backward Elimination işlemi için başlangıçta tüm değişkenlerle model oluşturuluyor
def backward_elimination(X, y, significance_level=0.05):
    # Başlangıçta X'e sabit terim ekleniyor
    X_with_intercept = sm.add_constant(X)
    columns = list(range(X_with_intercept.shape[1]))  # Başlangıçta tüm sütunların indeksleri
    # İlk modelin kurulması
    model = sm.OLS(y, X_with_intercept).fit()
    
    # P-değerlerini alıyoruz
    p_values = model.pvalues
    
    # Backward elimination: P-değeri en büyük olan değişkeni çıkarıyoruz
    while max(p_values) > significance_level:
        max_p_value_index = np.argmax(p_values)  # En yüksek p-değerini bulan index
        del columns[max_p_value_index] #columns.pop(max_p_value_index)  # O özelliği çıkarıyoruz
        X_with_intercept =  np.delete(X_with_intercept, max_p_value_index, axis=1)  # En yüksek p-değerine sahip değişkeni çıkarıyoruz
        model = sm.OLS(y, X_with_intercept).fit()  # Modeli tekrar kuruyoruz
        p_values = model.pvalues  # Yeni p-değerlerini alıyoruz
    
    selected_features = [i - 1 for i in columns if i > 0]
    print(f"Seçilen sütunlar (0 tabanlı indeksleme): {selected_features}")
    # Sonuçları döndürüyoruz
    return model, selected_features
#Backward elimination X2,X3 ve X5 özelliklerini çıkarttırdı p değerlerinden dolayı ama kalan özelliklerde korelasyon yüksek olarak devam etti. Yani işe yaramadı.
"""

# VIF hesaplama fonksiyonu
def calculate_vif(X):
    vif = pd.DataFrame()
    vif["Feature"] = [f"X{i+1}" for i in range(X.shape[1])]
    vif["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    return vif 

# Veri yükleme
data = pd.read_excel('GR10_Prediction.xlsx', sheet_name='Data')

# Eksik verileri doldur
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
dataArray = data.values
imputer = imputer.fit(dataArray)
dataArray = imputer.transform(dataArray)
dataArray = pd.DataFrame(data=dataArray, columns=data.columns)

# Aykırı Değerleri Tespit Etme ve Temizleme (Z-skoru ile)
z_scores = np.abs(zscore(dataArray))  # Z-skorunu hesapla
threshold = 3  # Z-skoru 3'ten büyükse aykırı kabul edilir
data_no_outliers = dataArray[(z_scores < threshold).all(axis=1)]


# Eğitim ve test setlerine ayırma
y = data_no_outliers.iloc[:, 8:9].values  # Y hedef değişkeni
X = data_no_outliers.iloc[:, 0:8].values  # X özellikler
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Veriyi standartlaştırma
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Korelasyon Matrisini Hesaplama # X Özelliklerinin Korelasyonu
correlation_matrix_x = pd.DataFrame(X, columns=data.columns[:8]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_x, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("X Özelliklerinin Korelasyon Matrisi")
plt.show()

# X Özelliklerinin Y'ye Olan Korelasyonu
y_corr = pd.DataFrame(X, columns=data.columns[:8]).apply(lambda x: np.corrcoef(x, y.flatten())[0, 1])
print("Y'ye olan korelasyonlar:")
print(y_corr)
plt.figure(figsize=(8, 6))
sns.barplot(x=y_corr.index, y=y_corr.values, palette='viridis', hue=y_corr.index, dodge=False)
plt.title('X Özelliklerinin Y ile Korelasyonu')
plt.xlabel('Özellikler')
plt.ylabel('Korelasyon Katsayısı')
plt.xticks(rotation=45)
plt.show()

X_with_intercept = sm.add_constant(X_train)  # Sabit terim ekliyoruz
vif_data = calculate_vif(X_with_intercept)
print("Başlangıç VIF Değerleri:")
print(vif_data)

# Orijinal özellik isimlerini başlangıçta al
original_feature_names = data.columns[:8].tolist()

# VIF eşiği kullanarak yüksek VIF değerine sahip özellikleri çıkar
vif_threshold = 5  # VIF eşik değeri (genelde 5 veya 10 kullanılır)
while vif_data["VIF"].max() > vif_threshold:
    max_vif_feature_index = vif_data["VIF"].idxmax()  # En yüksek VIF özelliğinin indeksi
    max_vif_feature_name = vif_data.loc[max_vif_feature_index, "Feature"]  # Özelliğin ismi (X1, X2 gibi)

    # Orijinal isimden çıkarılan özelliği bul
    removed_feature_name = original_feature_names[max_vif_feature_index - 1]  # -1 sabit terimi hesaba katmak için

    print(f"\nÇıkarılan Özellik: {removed_feature_name} (VIF: {vif_data['VIF'].max()})")

    # Özelliği orijinal isimler listesinden kaldır
    original_feature_names.pop(max_vif_feature_index - 1)

    # Özelliği X_train ve X_test'ten kaldır
    X_train = np.delete(X_train, max_vif_feature_index - 1, axis=1)  # Sabit terimi hesaba katmak için -1
    X_with_intercept = sm.add_constant(X_train)  # Yeniden sabit terim ekle

    X_test = np.delete(X_test, max_vif_feature_index - 1, axis=1)
    X_test_with_intercept = sm.add_constant(X_test)

    # Yeni VIF değerlerini hesapla
    vif_data = calculate_vif(X_with_intercept)
    print("Güncellenmiş VIF Değerleri:")
    print(vif_data)

# Son haliyle kalan özellikler
print("\nSon Kalan Özellikler (Orijinal İsimleriyle):")
print(original_feature_names)


# X_train'in kalan sütunlar üzerinden yeniden adlandırılması
X_train_df_corrected = pd.DataFrame(X_train, columns=original_feature_names)

# Korelasyon matrisi oluşturma ve görselleştirme
correlation_matrix = X_train_df_corrected.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("VIF Sonrası Kalan Özelliklerin Korelasyon Matrisi")
plt.show()

# Yeni bir model kurma
final_model = sm.OLS(y_train, X_with_intercept).fit()
print(final_model.summary())

y_pred = final_model.predict(X_test_with_intercept)  # Tahminler

# Model performansını değerlendirme
mae = mean_absolute_error(y_test, y_pred)  # Ortalama mutlak hata
mse = mean_squared_error(y_test, y_pred)  # Ortalama kare hata
r2 = r2_score(y_test, y_pred)  # R^2 skoru

print(f"Ortalama Mutlak Hata (MAE): {mae}")
print(f"Ortalama Kare Hata (MSE): {mse}")
print(f"R^2 Skoru: {r2}")

# Gerçek ve tahmin edilen değerleri karşılaştıran scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label="Tahminler")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linewidth=2, linestyle='--', label="Y = x (Gerçek)")
plt.title("Gerçek ve Tahmin Edilen Değerler", fontsize=16)
plt.xlabel("Gerçek Y Değerleri", fontsize=14)
plt.ylabel("Tahmin Edilen Y Değerleri", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

"""
# Korelasyonlar
positive_corr_features = ['X1', 'X5']  # Y'ye pozitif korelasyonlu özellikler
negative_corr_features = ['X2', 'X4']  # Y'ye negatif korelasyonlu özellikler

# Özellikleri X veri kümesinden alıyoruz
positive_corr_X = pd.DataFrame(X, columns=data.columns[:8])[positive_corr_features]
negative_corr_X = pd.DataFrame(X, columns=data.columns[:8])[negative_corr_features]

# Birleştirme: X1 ve X5'i, X2 ve X4'ü birleştiriyoruz
combined_positive_features = positive_corr_X.sum(axis=1)  # X1 ve X5'i toplama
combined_negative_features = negative_corr_X.sum(axis=1)  # X2 ve X4'ü toplama

# Yeni özellikleri oluşturuyoruz
X_combined = pd.DataFrame({
    'combined_positive': combined_positive_features,
    'combined_negative': combined_negative_features
})

# Eğitim ve test setlerine ayıralım
x_train_combined, x_test_combined, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=0)

# Veriyi standartlaştırma
sc = StandardScaler()
X_train_combined = sc.fit_transform(x_train_combined)
X_test_combined = sc.transform(x_test_combined)

# Model Kurma
model = LinearRegression()
model.fit(X_train_combined, y_train)

# Tahmin yapma
y_pred = model.predict(X_test_combined)

# Performans değerlendirmesi
r2 = r2_score(y_test, y_pred)
print("Modelin R^2 değeri:", r2)

# Gerçek ve tahmin edilen değerlerin karşılaştırılması
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label="Tahminler")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label="Y = x (Gerçek)")
plt.title("Gerçek ve Tahmin Edilen Değerler")
plt.xlabel("Gerçek Y Değerleri")
plt.ylabel("Tahmin Edilen Y Değerleri")
plt.legend()
plt.grid(True)
plt.show()"""


""" # direkt model kurma 

# Yeni model kurma 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Modelin katsayıları ve sabit terimi
print("\nModelin Katsayıları (Coefficients):")
print(regressor.coef_)

print("\nModelin Sabit Terimi (Intercept):")
print(regressor.intercept_)

# Test setinde modelin tahminlerini yapıyoruz
y_pred = regressor.predict(X_test)

# Performans metriklerini hesaplıyoruz
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Sonuçları yazdıralım
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

# Gerçek ve tahmin edilen değerleri karşılaştıran bir scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label="Tahminler")

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label="Y = x (Gerçek)")
plt.title("Gerçek ve Tahmin Edilen Değerler")
plt.xlabel("Gerçek Y Değerleri")
plt.ylabel("Tahmin Edilen Y Değerleri")
plt.legend()
plt.grid(True)
plt.show()
"""







"""
# PCA'yı uygulamak
pca = PCA(n_components=0.95)  # 0.95, toplam varyansın %95'ini açıklayacak bileşen sayısını seçer
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# PCA ile oluşturulan bileşenlerin sayısını kontrol et
print(f"Toplam bileşen sayısı: {X_train_pca.shape[1]}")

# PCA'dan gelen bileşenlerle model kurma
final_model_pca = sm.OLS(y_train, sm.add_constant(X_train_pca)).fit()

# Sonuçları yazdırma
print(final_model_pca.summary())

# Model ile tahmin yapma
y_pred_pca = final_model_pca.predict(sm.add_constant(X_test_pca))

# Sonuçları görselleştirme (gerçek ve tahmin edilen değerleri karşılaştırma)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_pca, color='blue', alpha=0.6, label="Tahminler")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label="Y = x (Gerçek)")
plt.title("Gerçek ve Tahmin Edilen Değerler (PCA Sonrası Model)")
plt.xlabel("Gerçek Y Değerleri")
plt.ylabel("Tahmin Edilen Y Değerleri")
plt.legend()
plt.grid(True)
plt.show()

"""








"""
#------ VIF KULLANARAK R^2 = 0.76 
# VIF değerlerini hesapla
X_with_intercept = sm.add_constant(X_train)  # Sabit terim ekliyoruz
vif_data = calculate_vif(X_with_intercept)
print("Başlangıç VIF Değerleri:")
print(vif_data)

# Yüksek VIF değerine sahip özellikleri çıkarmaya başla
vif_threshold = 5  # VIF eşik değeri (genelde 5 veya 10 kullanılır)
while vif_data["VIF"].max() > vif_threshold:
    max_vif_feature_index = vif_data["VIF"].idxmax()  # En yüksek VIF özelliğinin indeksi
    max_vif_feature_name = vif_data.loc[max_vif_feature_index, "Feature"]  # Özelliğin ismi

    print(f"\nÇıkarılan Özellik: {max_vif_feature_name}, VIF: {vif_data['VIF'].max()}")
    
    # Özelliği X_train'den çıkar
    X_train = np.delete(X_train, max_vif_feature_index - 1, axis=1)  # Sabit terimi hesaba katmak için -1
    X_with_intercept = sm.add_constant(X_train)  # Yeniden sabit terim ekle
    
    X_test = np.delete(X_test, max_vif_feature_index - 1, axis=1)
    X_test_with_intercept = sm.add_constant(X_test)
    
    # Yeni VIF değerlerini hesapla
    vif_data = calculate_vif(X_with_intercept)
    print("Güncellenmiş VIF Değerleri:")
    print(vif_data)

# Son haliyle X_train ve VIF tablosu
print("\nSon Özellik Seti:")
print(vif_data)

# Yeni bir model kurma
final_model = sm.OLS(y_train, X_with_intercept).fit()
print(final_model.summary())

y_pred = final_model.predict(X_test_with_intercept)  # Tahminler

# Model performansını değerlendirme
mae = mean_absolute_error(y_test, y_pred)  # Ortalama mutlak hata
mse = mean_squared_error(y_test, y_pred)  # Ortalama kare hata
r2 = r2_score(y_test, y_pred)  # R^2 skoru

print(f"Ortalama Mutlak Hata (MAE): {mae}")
print(f"Ortalama Kare Hata (MSE): {mse}")
print(f"R^2 Skoru: {r2}")

# Gerçek ve tahmin edilen değerleri karşılaştıran scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label="Tahminler")

# Y = x doğrusu (gerçek ve tahminin eşit olduğu ideal doğrultu)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linewidth=2, linestyle='--', label="Y = x (Gerçek)")

# Başlık, etiketler ve gösterge
plt.title("Gerçek ve Tahmin Edilen Değerler", fontsize=16)
plt.xlabel("Gerçek Y Değerleri", fontsize=14)
plt.ylabel("Tahmin Edilen Y Değerleri", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# Görselleştirme
plt.tight_layout()
plt.show()
"""



"""
#Backward Elimination r^2 DEĞERİ =0.9 GİBİ BİR ŞEYDİ
final_model, selected_features = backward_elimination (X_train, y_train)
print(final_model.summary())

#Backward Eliminaitondan gelen değerlerle model kur !! 

# Backward Elimination'dan seçilen özelliklere göre yeni X verisini oluşturma
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

columns = [f'X{i+1}' for i in selected_features]  # Seçilen sütunların isimlendirilmesi
X_train_selected_df = pd.DataFrame(X_train_selected, columns=columns)

# Korelasyon matrisini hesapla
correlation_matrix = X_train_selected_df.corr()

# Korelasyon matrisini görselleştir
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Backward Sonrası Seçilen Özelliklerin Korelasyon Matrisi", fontsize=16)
plt.show()

# Yeni modelin oluşturulması ve eğitilmesi
regressor = LinearRegression()  # Çoklu doğrusal regresyon modeli
regressor.fit(X_train_selected, y_train)

# Tahminler
y_pred = regressor.predict(X_test_selected)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R^2 Skoru: {r2}")
print(f"Ortalama Kare Hatası (MSE): {mse}")
print(f"Ortalama Mutlak Hata (MAE): {mae}")

# Tahmin ve gerçek değerlerin karşılaştırılması
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label="Tahminler")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label="Y = x (Gerçek)")
plt.title("Backward Sonrası Gerçek ve Tahmin Edilen Değerler")
plt.xlabel("Gerçek Y Değerleri")
plt.ylabel("Tahmin Edilen Y Değerleri")
plt.legend()
plt.grid(True)
plt.show() """



