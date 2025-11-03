"""
Zara Sales Data Analysis - Complete ML/DL Project
Author: Senior ML/DL Developer
Description: End-to-end machine learning pipeline with data cleaning, EDA, and modeling
"""

# ============================================================================
# SECTION 1: IMPORTS AND SETUP
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
pd.set_option('display.max_columns', None)

print("="*80)
print("ZARA SALES DATA ANALYSIS - ML PROJECT WITH PCA")
print("="*80)

# ============================================================================
# SECTION 2: LOAD DATA
# ============================================================================

print("Loading dataset...")
df = pd.read_csv('Zara_sales_EDA.csv', delimiter=';')
print(f"✓ Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# ============================================================================
# SECTION 3: INITIAL EXPLORATION
# ============================================================================

print("="*80)
print("INITIAL DATA EXPLORATION")
print("="*80)
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nFirst few rows:")
print(df.head())

# ============================================================================
# SECTION 4: MISSING VALUES ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("MISSING VALUES ANALYSIS")
print("="*80)

missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum().values,
    'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_data) > 0:
    print(missing_data)
    
    # Handle missing values
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    print("\n✓ Missing values handled!")
else:
    print("✓ No missing values found!")

print(f"Remaining missing values: {df.isnull().sum().sum()}")

# Check duplicates
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")


# ============================================================================
# SECTION 5: UNIVARIATE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("UNIVARIATE ANALYSIS")
print("="*80)

# Numerical features
numerical_features = ['Sales Volume', 'price']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Univariate Analysis - Numerical Features', fontsize=16, fontweight='bold')

for idx, col in enumerate(numerical_features):
    # Histogram
    axes[idx, 0].hist(df[col], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[idx, 0].set_title(f'{col} - Distribution')
    axes[idx, 0].set_xlabel(col)
    axes[idx, 0].set_ylabel('Frequency')
    axes[idx, 0].axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.2f}')
    axes[idx, 0].axvline(df[col].median(), color='green', linestyle='--', label=f'Median: {df[col].median():.2f}')
    axes[idx, 0].legend()
    axes[idx, 0].grid(True, alpha=0.3)
    
    # Box plot
    axes[idx, 1].boxplot(df[col], vert=True, patch_artist=True)
    axes[idx, 1].set_title(f'{col} - Box Plot')
    axes[idx, 1].set_ylabel(col)
    axes[idx, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('univariate_numerical.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nNumerical Features Statistics:")
print(df[numerical_features].describe())

# Categorical features
categorical_features = ['Product Position', 'Promotion', 'Seasonal', 'section', 'season', 'material', 'origin']

fig, axes = plt.subplots(4, 2, figsize=(16, 20))
fig.suptitle('Univariate Analysis - Categorical Features', fontsize=16, fontweight='bold')

for idx, col in enumerate(categorical_features):
    row = idx // 2
    col_idx = idx % 2
    
    value_counts = df[col].value_counts().head(10)
    axes[row, col_idx].barh(value_counts.index.astype(str), value_counts.values, color='coral')
    axes[row, col_idx].set_title(f'{col} - Top 10 Categories')
    axes[row, col_idx].set_xlabel('Count')
    axes[row, col_idx].grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(value_counts.values):
        axes[row, col_idx].text(v, i, f' {v}', va='center')

plt.tight_layout()
plt.savefig('univariate_categorical.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nCategorical Features Value Counts:")
for col in categorical_features[:3]:  # Print first 3 to save space
    print(f"\n{col}:")
    print(df[col].value_counts().head(5))


# ============================================================================
# SECTION 6: BIVARIATE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("BIVARIATE ANALYSIS")
print("="*80)

# Correlation analysis
numerical_df = df.select_dtypes(include=[np.number])
correlation_matrix = numerical_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.2f', cbar_kws={'label': 'Correlation'})
plt.title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nCorrelation with Sales Volume:")
if 'Sales Volume' in correlation_matrix.columns:
    print(correlation_matrix['Sales Volume'].sort_values(ascending=False))

# Sales Volume vs Price
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].scatter(df['price'], df['Sales Volume'], alpha=0.5, color='blue')
axes[0].set_xlabel('Price (USD)')
axes[0].set_ylabel('Sales Volume')
axes[0].set_title('Sales Volume vs Price - Scatter Plot')
axes[0].grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(df['price'], df['Sales Volume'], 1)
p = np.poly1d(z)
axes[0].plot(df['price'].sort_values(), p(df['price'].sort_values()), "r--", alpha=0.8, linewidth=2, label='Trend Line')
axes[0].legend()

# Hexbin plot
axes[1].hexbin(df['price'], df['Sales Volume'], gridsize=30, cmap='YlOrRd')
axes[1].set_xlabel('Price (USD)')
axes[1].set_ylabel('Sales Volume')
axes[1].set_title('Sales Volume vs Price - Density Plot')
plt.colorbar(axes[1].collections[0], ax=axes[1], label='Count')

plt.tight_layout()
plt.savefig('bivariate_price_sales.png', dpi=300, bbox_inches='tight')
plt.show()

# Categorical vs Sales Volume
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Sales Volume by Categorical Features', fontsize=16, fontweight='bold')

# Promotion vs Sales Volume
promotion_sales = df.groupby('Promotion')['Sales Volume'].mean().sort_values(ascending=False)
axes[0, 0].bar(promotion_sales.index.astype(str), promotion_sales.values, color=['#FF6B6B', '#4ECDC4'])
axes[0, 0].set_title('Average Sales Volume by Promotion')
axes[0, 0].set_xlabel('Promotion')
axes[0, 0].set_ylabel('Average Sales Volume')
axes[0, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(promotion_sales.values):
    axes[0, 0].text(i, v, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')

# Product Position vs Sales Volume
position_sales = df.groupby('Product Position')['Sales Volume'].mean().sort_values(ascending=False)
axes[0, 1].bar(range(len(position_sales)), position_sales.values, color='skyblue')
axes[0, 1].set_xticks(range(len(position_sales)))
axes[0, 1].set_xticklabels(position_sales.index, rotation=45, ha='right')
axes[0, 1].set_title('Average Sales Volume by Product Position')
axes[0, 1].set_ylabel('Average Sales Volume')
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(position_sales.values):
    axes[0, 1].text(i, v, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')

# Season vs Sales Volume
season_sales = df.groupby('season')['Sales Volume'].mean().sort_values(ascending=False)
axes[1, 0].bar(range(len(season_sales)), season_sales.values, color='lightcoral')
axes[1, 0].set_xticks(range(len(season_sales)))
axes[1, 0].set_xticklabels(season_sales.index, rotation=45, ha='right')
axes[1, 0].set_title('Average Sales Volume by Season')
axes[1, 0].set_ylabel('Average Sales Volume')
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(season_sales.values):
    axes[1, 0].text(i, v, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')

# Section vs Sales Volume
section_sales = df.groupby('section')['Sales Volume'].mean().sort_values(ascending=False)
axes[1, 1].bar(range(len(section_sales)), section_sales.values, color='lightgreen')
axes[1, 1].set_xticks(range(len(section_sales)))
axes[1, 1].set_xticklabels(section_sales.index, rotation=45, ha='right')
axes[1, 1].set_title('Average Sales Volume by Section')
axes[1, 1].set_ylabel('Average Sales Volume')
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(section_sales.values):
    axes[1, 1].text(i, v, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('bivariate_categorical.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Bivariate analysis completed!")


# ============================================================================
# SECTION 7: FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

# Create new features
print("Creating new features...")

# 1. Price categories
df['price_category'] = pd.cut(df['price'], bins=[0, 25, 50, 75, 100, np.inf], 
                               labels=['Budget', 'Mid-Range', 'Premium', 'Luxury', 'Ultra-Luxury'])

# 2. Sales categories
df['sales_category'] = pd.cut(df['Sales Volume'], bins=[0, 800, 1200, 1600, 2000], 
                               labels=['Low', 'Medium', 'High', 'Very High'])

# 3. Seasonal indicators
df['is_winter'] = (df['season'] == 'Winter').astype(int)
df['is_summer'] = (df['season'] == 'Summer').astype(int)
df['is_autumn'] = (df['season'] == 'Autumn').astype(int)
df['is_spring'] = (df['season'] == 'Spring').astype(int)

# 4. Price per sales ratio
df['price_sales_ratio'] = df['price'] / (df['Sales Volume'] + 1)

print("✓ New features created!")
print(f"New columns: price_category, sales_category, seasonal indicators, price_sales_ratio")

# Visualize new features
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Feature Engineering Visualizations', fontsize=16, fontweight='bold')

# Price category distribution
price_cat_counts = df['price_category'].value_counts()
axes[0, 0].pie(price_cat_counts.values, labels=price_cat_counts.index, autopct='%1.1f%%', 
               startangle=90, colors=sns.color_palette('Set2'))
axes[0, 0].set_title('Price Category Distribution')

# Sales category distribution
sales_cat_counts = df['sales_category'].value_counts()
axes[0, 1].pie(sales_cat_counts.values, labels=sales_cat_counts.index, autopct='%1.1f%%', 
               startangle=90, colors=sns.color_palette('Set3'))
axes[0, 1].set_title('Sales Category Distribution')

# Price vs Sales colored by promotion
for promo in df['Promotion'].unique():
    mask = df['Promotion'] == promo
    axes[1, 0].scatter(df[mask]['price'], df[mask]['Sales Volume'], 
                      label=f'Promotion: {promo}', alpha=0.6, s=30)
axes[1, 0].set_xlabel('Price (USD)')
axes[1, 0].set_ylabel('Sales Volume')
axes[1, 0].set_title('Price vs Sales Volume (by Promotion)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Top 10 origins by average sales
top_origins = df.groupby('origin')['Sales Volume'].mean().sort_values(ascending=False).head(10)
axes[1, 1].barh(range(len(top_origins)), top_origins.values, color='teal')
axes[1, 1].set_yticks(range(len(top_origins)))
axes[1, 1].set_yticklabels(top_origins.index)
axes[1, 1].set_xlabel('Average Sales Volume')
axes[1, 1].set_title('Top 10 Origins by Average Sales')
axes[1, 1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(top_origins.values):
    axes[1, 1].text(v, i, f' {v:.0f}', va='center')

plt.tight_layout()
plt.savefig('feature_engineering.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# SECTION 8: DATA PREPROCESSING FOR ML
# ============================================================================

print("\n" + "="*80)
print("DATA PREPROCESSING FOR MACHINE LEARNING")
print("="*80)

# Create a copy for ML
df_ml = df.copy()

# Select features for encoding
features_to_encode = ['Product Position', 'Promotion', 'Seasonal', 'section', 'season', 'material', 'origin']
target = 'Sales Volume'

# Label encoding
label_encoders = {}
for col in features_to_encode:
    le = LabelEncoder()
    df_ml[col + '_encoded'] = le.fit_transform(df_ml[col].astype(str))
    label_encoders[col] = le

# Select features for modeling
feature_columns = ['price', 'Product Position_encoded', 'Promotion_encoded', 
                   'Seasonal_encoded', 'section_encoded', 'season_encoded', 
                   'material_encoded', 'origin_encoded']

X = df_ml[feature_columns]
y = df_ml[target]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Features: {feature_columns}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} ({(X_train.shape[0]/len(X)*100):.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({(X_test.shape[0]/len(X)*100):.1f}%)")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Data preprocessing completed!")


# ============================================================================
# SECTION 9: MACHINE LEARNING MODELS
# ============================================================================

print("\n" + "="*80)
print("MACHINE LEARNING MODELS")
print("="*80)

ml_results = {}

# 1. Linear Regression
print("\n1. Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

ml_results['Linear Regression'] = {'RMSE': lr_rmse, 'MAE': lr_mae, 'R2': lr_r2}
print(f"   RMSE: {lr_rmse:.2f} | MAE: {lr_mae:.2f} | R²: {lr_r2:.4f}")

# 2. Ridge Regression
print("\n2. Training Ridge Regression...")
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_scaled, y_train)
ridge_pred = ridge_model.predict(X_test_scaled)

ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

ml_results['Ridge Regression'] = {'RMSE': ridge_rmse, 'MAE': ridge_mae, 'R2': ridge_r2}
print(f"   RMSE: {ridge_rmse:.2f} | MAE: {ridge_mae:.2f} | R²: {ridge_r2:.4f}")

# 3. Random Forest
print("\n3. Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

ml_results['Random Forest'] = {'RMSE': rf_rmse, 'MAE': rf_mae, 'R2': rf_r2}
print(f"   RMSE: {rf_rmse:.2f} | MAE: {rf_mae:.2f} | R²: {rf_r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\n   Top 5 Important Features:")
print(feature_importance.head())

# 4. Gradient Boosting
print("\n4. Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_mae = mean_absolute_error(y_test, gb_pred)
gb_r2 = r2_score(y_test, gb_pred)

ml_results['Gradient Boosting'] = {'RMSE': gb_rmse, 'MAE': gb_mae, 'R2': gb_r2}
print(f"   RMSE: {gb_rmse:.2f} | MAE: {gb_mae:.2f} | R²: {gb_r2:.4f}")

print("\n" + "="*80)
print("ML MODELS COMPARISON")
print("="*80)
results_df = pd.DataFrame(ml_results).T
print(results_df)


# ============================================================================
# SECTION 10: PRINCIPAL COMPONENT ANALYSIS (PCA)
# ============================================================================

print("\n" + "="*80)
print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("="*80)

# Apply PCA
print("\nApplying PCA to the scaled features...")

# PCA with all components to see variance explained
pca_full = PCA()
pca_full.fit(X_train_scaled)

# Calculate cumulative variance explained
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

print(f"\nTotal features: {X_train_scaled.shape[1]}")
print(f"Variance explained by each component:")
for i, var in enumerate(pca_full.explained_variance_ratio_):
    print(f"   PC{i+1}: {var*100:.2f}% (Cumulative: {cumulative_variance[i]*100:.2f}%)")

# Visualize variance explained
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Scree plot
axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1), 
            pca_full.explained_variance_ratio_ * 100, 
            color='steelblue', alpha=0.7)
axes[0].plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
             pca_full.explained_variance_ratio_ * 100, 
             'ro-', linewidth=2, markersize=8)
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained (%)')
axes[0].set_title('Scree Plot - Variance Explained by Each PC')
axes[0].grid(True, alpha=0.3)

# Cumulative variance plot
axes[1].plot(range(1, len(cumulative_variance) + 1), 
             cumulative_variance * 100, 
             'bo-', linewidth=2, markersize=8)
axes[1].axhline(y=95, color='r', linestyle='--', label='95% Variance')
axes[1].axhline(y=90, color='orange', linestyle='--', label='90% Variance')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance Explained (%)')
axes[1].set_title('Cumulative Variance Explained')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_variance_explained.png', dpi=300, bbox_inches='tight')
plt.show()

# Determine optimal number of components (95% variance)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nComponents needed for 95% variance: {n_components_95}")

# Apply PCA with optimal components
pca = PCA(n_components=n_components_95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Original feature space: {X_train_scaled.shape[1]} dimensions")
print(f"Reduced feature space: {X_train_pca.shape[1]} dimensions")
print(f"Dimensionality reduction: {((1 - X_train_pca.shape[1]/X_train_scaled.shape[1])*100):.1f}%")

# Visualize first 2 principal components
if X_train_pca.shape[1] >= 2:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                         c=y_train, cmap='viridis', alpha=0.6, s=20)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.title('Training Data - First 2 Principal Components')
    plt.colorbar(scatter, label='Sales Volume')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], 
                         c=y_test, cmap='viridis', alpha=0.6, s=20)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.title('Test Data - First 2 Principal Components')
    plt.colorbar(scatter, label='Sales Volume')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_2d_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

# Component loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=feature_columns
)

print("\nPCA Component Loadings (Top 3 PCs):")
print(loadings.iloc[:, :min(3, pca.n_components_)].round(3))

# Visualize loadings for first 2 components
if pca.n_components_ >= 2:
    plt.figure(figsize=(10, 6))
    plt.scatter(loadings['PC1'], loadings['PC2'], s=100, alpha=0.6)
    
    for i, feature in enumerate(feature_columns):
        plt.annotate(feature, (loadings['PC1'][i], loadings['PC2'][i]), 
                    fontsize=9, alpha=0.8)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.title('PCA Loading Plot - Feature Contributions')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_loadings.png', dpi=300, bbox_inches='tight')
    plt.show()

# Train models with PCA-transformed data
print("\n" + "="*80)
print("TRAINING MODELS WITH PCA FEATURES")
print("="*80)

pca_results = {}

# Random Forest with PCA
print("\nTraining Random Forest with PCA features...")
rf_pca = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_pca.fit(X_train_pca, y_train)
rf_pca_pred = rf_pca.predict(X_test_pca)

rf_pca_rmse = np.sqrt(mean_squared_error(y_test, rf_pca_pred))
rf_pca_mae = mean_absolute_error(y_test, rf_pca_pred)
rf_pca_r2 = r2_score(y_test, rf_pca_pred)

pca_results['Random Forest (PCA)'] = {'RMSE': rf_pca_rmse, 'MAE': rf_pca_mae, 'R2': rf_pca_r2}
print(f"   RMSE: {rf_pca_rmse:.2f} | MAE: {rf_pca_mae:.2f} | R²: {rf_pca_r2:.4f}")

# Gradient Boosting with PCA
print("\nTraining Gradient Boosting with PCA features...")
gb_pca = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb_pca.fit(X_train_pca, y_train)
gb_pca_pred = gb_pca.predict(X_test_pca)

gb_pca_rmse = np.sqrt(mean_squared_error(y_test, gb_pca_pred))
gb_pca_mae = mean_absolute_error(y_test, gb_pca_pred)
gb_pca_r2 = r2_score(y_test, gb_pca_pred)

pca_results['Gradient Boosting (PCA)'] = {'RMSE': gb_pca_rmse, 'MAE': gb_pca_mae, 'R2': gb_pca_r2}
print(f"   RMSE: {gb_pca_rmse:.2f} | MAE: {gb_pca_mae:.2f} | R²: {gb_pca_r2:.4f}")

print("\n✓ PCA analysis completed!")


# ============================================================================
# SECTION 11: MODEL COMPARISON AND VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("FINAL MODEL COMPARISON")
print("="*80)

# Combine all results
all_results = {**ml_results, **pca_results}
final_results = pd.DataFrame(all_results).T.sort_values('R2', ascending=False)
print(final_results)

# Compare original vs PCA models
print("\n" + "="*80)
print("ORIGINAL vs PCA FEATURES COMPARISON")
print("="*80)
print("\nOriginal Features:")
print(pd.DataFrame(ml_results).T)
print("\nPCA Features:")
print(pd.DataFrame(pca_results).T)

# Visualize model comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Performance Comparison (All Models)', fontsize=16, fontweight='bold')

models = final_results.index
metrics = ['RMSE', 'MAE', 'R2']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F4A460']

for idx, metric in enumerate(metrics):
    axes[idx].bar(range(len(models)), final_results[metric].values, color=colors[:len(models)])
    axes[idx].set_xticks(range(len(models)))
    axes[idx].set_xticklabels(models, rotation=45, ha='right')
    axes[idx].set_title(f'{metric} Comparison')
    axes[idx].set_ylabel(metric)
    axes[idx].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(final_results[metric].values):
        axes[idx].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Predictions comparison
predictions = {
    'Linear Regression': lr_pred,
    'Ridge Regression': ridge_pred,
    'Random Forest': rf_pred,
    'Gradient Boosting': gb_pred,
    'Random Forest (PCA)': rf_pca_pred,
    'Gradient Boosting (PCA)': gb_pca_pred
}

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Actual vs Predicted Sales Volume', fontsize=16, fontweight='bold')

for idx, (model_name, pred) in enumerate(predictions.items()):
    row = idx // 3
    col = idx % 3
    
    axes[row, col].scatter(y_test, pred, alpha=0.5, s=20)
    axes[row, col].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                        'r--', lw=2, label='Perfect Prediction')
    axes[row, col].set_xlabel('Actual Sales Volume')
    axes[row, col].set_ylabel('Predicted Sales Volume')
    r2_val = all_results[model_name]["R2"]
    axes[row, col].set_title(f'{model_name}\nR² = {r2_val:.4f}')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Residual plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Residual Plots', fontsize=16, fontweight='bold')

for idx, (model_name, pred) in enumerate(predictions.items()):
    row = idx // 3
    col = idx % 3
    
    residuals = y_test - pred
    axes[row, col].scatter(pred, residuals, alpha=0.5, s=20)
    axes[row, col].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[row, col].set_xlabel('Predicted Sales Volume')
    axes[row, col].set_ylabel('Residuals')
    axes[row, col].set_title(f'{model_name}')
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance visualization
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importance)), feature_importance['Importance'].values, color='steelblue')
plt.yticks(range(len(feature_importance)), feature_importance['Feature'].values)
plt.xlabel('Importance Score')
plt.title('Feature Importance - Random Forest Model', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# SECTION 12: FINAL SUMMARY AND INSIGHTS
# ============================================================================

print("\n" + "="*80)
print("PROJECT SUMMARY")
print("="*80)

best_model = final_results.index[0]
print(f"\n🏆 BEST PERFORMING MODEL: {best_model}")
print(f"   R² Score: {final_results.loc[best_model, 'R2']:.4f}")
print(f"   RMSE: {final_results.loc[best_model, 'RMSE']:.2f}")
print(f"   MAE: {final_results.loc[best_model, 'MAE']:.2f}")

print("\n" + "="*80)
print("KEY INSIGHTS FROM ANALYSIS")
print("="*80)

print("\n📊 Data Insights:")
print(f"   • Total products analyzed: {len(df)}")
print(f"   • Average sales volume: {df['Sales Volume'].mean():.0f}")
print(f"   • Average price: ${df['price'].mean():.2f}")
print(f"   • Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")

print("\n🎯 Sales Insights:")
promotion_impact = df.groupby('Promotion')['Sales Volume'].mean()
print(f"   • Products with promotion: {promotion_impact.get('Yes', 0):.0f} avg sales")
print(f"   • Products without promotion: {promotion_impact.get('No', 0):.0f} avg sales")

position_best = df.groupby('Product Position')['Sales Volume'].mean().idxmax()
print(f"   • Best performing position: {position_best}")

season_best = df.groupby('season')['Sales Volume'].mean().idxmax()
print(f"   • Best performing season: {season_best}")

print("\n🔍 Model Performance:")
print(f"   • Total models trained: {len(all_results)}")
print(f"   • Best R² score: {final_results['R2'].max():.4f}")
print(f"   • Lowest RMSE: {final_results['RMSE'].min():.2f}")

print("\n�  PCA Insights:")
print(f"   • Original dimensions: {X_train_scaled.shape[1]}")
print(f"   • Reduced dimensions: {X_train_pca.shape[1]}")
print(f"   • Variance retained: 95%")
print(f"   • Dimensionality reduction: {((1 - X_train_pca.shape[1]/X_train_scaled.shape[1])*100):.1f}%")

print("\n💡 Recommendations:")
print("   1. Focus on products with promotions for higher sales")
print(f"   2. Optimize product placement in '{position_best}' positions")
print(f"   3. Increase inventory for '{season_best}' season")
print("   4. Consider price optimization based on feature importance")
print(f"   5. Use {best_model} for sales predictions")
print(f"   6. PCA can reduce features by {((1 - X_train_pca.shape[1]/X_train_scaled.shape[1])*100):.1f}% with minimal performance loss")

print("\n" + "="*80)
print("✅ COMPLETE ML PROJECT WITH PCA FINISHED SUCCESSFULLY!")
print("="*80)

print("\n📁 Generated Files:")
print("   • univariate_numerical.png")
print("   • univariate_categorical.png")
print("   • correlation_matrix.png")
print("   • bivariate_price_sales.png")
print("   • bivariate_categorical.png")
print("   • feature_engineering.png")
print("   • pca_variance_explained.png")
print("   • pca_2d_visualization.png")
print("   • pca_loadings.png")
print("   • model_comparison.png")
print("   • predictions_comparison.png")
print("   • residual_plots.png")
print("   • feature_importance.png")

print("\n🎓 Project Components Completed:")
print("   ✓ Data Loading & Exploration")
print("   ✓ Data Cleaning & Missing Values Handling")
print("   ✓ Univariate Analysis")
print("   ✓ Bivariate Analysis")
print("   ✓ Feature Engineering")
print("   ✓ Data Preprocessing")
print("   ✓ Principal Component Analysis (PCA)")
print("   ✓ Machine Learning Models (4 models)")
print("   ✓ PCA-based Models (2 models)")
print("   ✓ Model Evaluation & Comparison")
print("   ✓ Visualizations & Insights")

print("\n" + "="*80)
