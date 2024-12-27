# -*- coding: utf-8 -*-
import optuna
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
from sklearn.ensemble import RandomForestRegressor
#from sklearn.decomposition import PCA

def objective(trial):
    # Suggest hyperparameters using Optuna
    n_estimators = trial.suggest_int('n_estimators', 10, 300)
    max_depth = trial.suggest_int('max_depth', 5, 50, step=5)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 6)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    # Create the model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    # Evaluate performance using cross-validation
    model.fit(X_train, y_train.flatten())
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse  # The objective functon aims to minimize MSE

# Function to calculate VIF
def calculate_vif(X):
    vif = pd.DataFrame()
    vif["Feature"] = [f"X{i+1}" for i in range(X.shape[1])]
    vif["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    return vif 

# Load the data 
data = pd.read_excel('GR10_Prediction.xlsx', sheet_name='Data')

# Handle missing values
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
dataArray = data.values
imputer = imputer.fit(dataArray)
dataArray = imputer.transform(dataArray)
dataArray = pd.DataFrame(data=dataArray, columns=data.columns)

# Detect and clean outliers using Z-score
z_scores = np.abs(zscore(dataArray))  # Calculate Z-score
threshold = 3  # Z-score greater than 3 is considered an outlier
data_no_outliers = dataArray[(z_scores < threshold).all(axis=1)]


# Split the data into training and testing sets
y = data_no_outliers.iloc[:, 8:9].values  # Target variable Y
X = data_no_outliers.iloc[:, 0:8].values  # Features X
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# standardize the data 
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)



# Calculate correlation matrix for X features
correlation_matrix_x = pd.DataFrame(X, columns=data.columns[:8]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_x, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix of X Features")
plt.show()

# Correlation between X features and Y 
y_corr = pd.DataFrame(X, columns=data.columns[:8]).apply(lambda x: np.corrcoef(x, y.flatten())[0, 1])
print("Correlations of features X to Y")
print(y_corr)
plt.figure(figsize=(8, 6))
sns.barplot(x=y_corr.index, y=y_corr.values, palette='viridis', hue=y_corr.index, dodge=False)
plt.title('Correlations of features X to Y')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45)
plt.show()

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print results
print("Multiple Linear Regression")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

# Scatter plot to compare actual and predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label="Predictions ")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label="Y = x (Real)")
plt.title("Real and Predicted Values (Multiple Linear Regression)")
plt.xlabel("Real Y Values")
plt.ylabel("Predicted Y Values")
plt.legend()
plt.grid(True)
plt.show()

X_with_intercept = sm.add_constant(X_train)  # Add constant term
vif_data = calculate_vif(X_with_intercept)
print("Initial VIF values:")
print(vif_data)

# Get original feature names
original_feature_names = data.columns[:8].tolist()

# Remove features with high VIF using a threshold
vif_threshold = 5  # VIF threshold (commonly 5 or 10)
while vif_data["VIF"].max() > vif_threshold:
    max_vif_feature_index = vif_data["VIF"].idxmax()  # Index of the highest VIF feature
    max_vif_feature_name = vif_data.loc[max_vif_feature_index, "Feature"]

    # Find the removed feature from the original names
    removed_feature_name = original_feature_names[max_vif_feature_index - 1]  # -1 to account for constant term

    print(f"\nRemoved Feature: {removed_feature_name} (VIF: {vif_data['VIF'].max()})")

    # Remove the feature from the original names list
    original_feature_names.pop(max_vif_feature_index - 1)

    # Remove the feature from X_traind and X_test
    X_train = np.delete(X_train, max_vif_feature_index - 1, axis=1)  # -1 to account for constant term
    X_with_intercept = sm.add_constant(X_train)  # Add constant term again

    X_test = np.delete(X_test, max_vif_feature_index - 1, axis=1)
    X_test_with_intercept = sm.add_constant(X_test)

    # Revalvulate VIF
    vif_data = calculate_vif(X_with_intercept)
    print("Updated VIF Values:")
    print(vif_data)

# Remaining features after VIF
print("\nLast Remaining Features (With Their Original Names):")
print(original_feature_names)


# Rename columns of X_train to the remaining features
X_train_df_corrected = pd.DataFrame(X_train, columns=original_feature_names)

# Create and visualize the correlation matrix after VIF
correlation_matrix = X_train_df_corrected.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix of Remaining Features After VIF")
plt.show()

# Calculate the correlation between remaining features and Y after VIF
y_corr_vif = X_train_df_corrected.apply(lambda x: np.corrcoef(x, y_train.flatten())[0, 1])
plt.figure(figsize=(8, 6))
sns.barplot(x=y_corr_vif.index, y=y_corr_vif.values, palette='viridis', dodge=False)
plt.title('Correlations of Remaining Features After VIF to Y', fontsize=14)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Correlation Coefficient', fontsize=12)
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Create a new model
final_model = sm.OLS(y_train, X_with_intercept).fit()
print(final_model.summary())

y_pred = final_model.predict(X_test_with_intercept)   # Predictions

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)  
mse = mean_squared_error(y_test, y_pred)  
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("After VIF Method")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

# Scatter plot to compare actual and predicted values after VIF
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label="Prediction")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linewidth=2, linestyle='--', label="Y = x (Real)")
plt.title("Real and Predicted Values after VIF", fontsize=16)
plt.xlabel("Real Y Values", fontsize=14)
plt.ylabel("Predicted Y Values", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Start Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, n_jobs=-1)

# Retrieve the best hyperparameters
best_params = study.best_params
print("\nBest hyperparameters:", best_params)

# Make predictions with the best model
best_model = RandomForestRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train.flatten())
y_pred_best = best_model.predict(X_test)

# Performance metrics
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_pred_best)

print("\nTop Model Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae_best}")
print(f"Mean Squared Error (MSE): {mse_best}")
print(f"Root Mean Squared Error (RMSE): {rmse_best}")
print(f"R-squared (R²): {r2_best}")

#  Actual vs Prediction chart
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, color='green', alpha=0.6, label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='blue', linewidth=2, linestyle='--', label="Y = x (Real)")
plt.title("Real and Predicted Values (with Bayesian Optimization)", fontsize=16)
plt.xlabel("Real Y Values", fontsize=14)
plt.ylabel("Predicted Y Values", fontsize=14)
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



