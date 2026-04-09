import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

##LOAD & INSPECT DATA

df = pd.read_csv("Project Data.csv")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nInfo:")
df.info()
print("\nDescribe:")
print(df.describe())

df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], dayfirst=True)

##DATA CLEANING

print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Zero Values ---")
print((df == 0).sum())

# Replace 0 with NaN in price columns
df['Min_Price']   = df['Min_Price'].replace(0, np.nan)
df['Max_Price']   = df['Max_Price'].replace(0, np.nan)
df['Modal_Price'] = df['Modal_Price'].replace(0, np.nan)

# Fill with median
df['Min_Price']   = df['Min_Price'].fillna(df['Min_Price'].median())
df['Max_Price']   = df['Max_Price'].fillna(df['Max_Price'].median())
df['Modal_Price'] = df['Modal_Price'].fillna(df['Modal_Price'].median())

print("\n--- After Cleaning ---")
print(df.isnull().sum())

#-----------------------CROP SELECTION----------------------------

selected_crops = ['Wheat', 'Green Peas', 'Mustard', 'Gur(Jaggery)']
df_selected = df[df['Commodity'].isin(selected_crops)].copy()
print("\nSelected Crop Record Counts:")
print(df_selected['Commodity'].value_counts())

#---------------OUTLIER DETECTION — IQR METHOD-------------------


print("\n========== OUTLIER DETECTION: IQR Method ==========")

column = 'Modal_Price'
Q1 = df_selected[column].quantile(0.25)
Q3 = df_selected[column].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1: {Q1},  Q3: {Q3},  IQR: {IQR}")
print(f"Lower Bound: {lower_bound},  Upper Bound: {upper_bound}")

outliers_iqr = df_selected[
    (df_selected[column] < lower_bound) | (df_selected[column] > upper_bound)
]
print(f"Outliers detected (IQR): {len(outliers_iqr)}")
print(outliers_iqr[['Commodity', 'Market', 'Modal_Price']].head(10))

# Box plot for outlier visualisation 
plt.figure(figsize=(8, 5))
sns.boxplot(x=df_selected[column])
plt.title("Boxplot for Outlier Detection - Modal Price")
plt.xlabel("Modal Price")
plt.tight_layout()
plt.show()

# Remove outliers for clean analysis
df_clean = df_selected[
    (df_selected[column] >= lower_bound) & (df_selected[column] <= upper_bound)
].copy()
print(f"\nRows before outlier removal: {len(df_selected)}")
print(f"Rows after  outlier removal: {len(df_clean)}")

#--------------------OUTLIER DETECTION — Z-SCORE METHOD-------------------

print("\n========== OUTLIER DETECTION: Z-Score Method ==========")

prices = df_selected['Modal_Price'].tolist()
mean_price = np.mean(prices)
std_price  = np.std(prices)
threshold  = 3

print(f"Mean: {round(mean_price, 2)},  Std Dev: {round(std_price, 2)}")

outliers_zscore = []
for price in prices:
    z = (price - mean_price) / std_price
    if abs(z) > threshold:
        outliers_zscore.append(price)

print(f"Outliers detected (Z-score, threshold={threshold}): {len(outliers_zscore)}")
print("Sample outlier values:", outliers_zscore[:10])

#-----------------------COVARIANCE & CORRELATION------------------------------

print("\n========== COVARIANCE & CORRELATION ==========")

price_df = df_selected[['Min_Price', 'Max_Price', 'Modal_Price']]

print("\nCovariance Matrix:")
print(price_df.cov())

print("\nCorrelation Matrix:")
corr_matrix = price_df.corr()
print(corr_matrix)

# Heatmap - seaborn 
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap — Price Variables")
plt.tight_layout()
plt.show()

#----------------------OBJECTIVE 2 — PRICE DISTRIBUTION ANALYSIS-------------------


# --- Seaborn histplot ---
plt.figure(figsize=(10, 5))
sns.histplot(df_selected['Modal_Price'], bins=30, kde=True)
plt.title("Distribution of Modal Prices (Seaborn histplot)")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# --- Matplotlib histogram ---
plt.figure(figsize=(10, 5))
plt.hist(df_selected['Modal_Price'], bins=30, facecolor='steelblue', edgecolor='white')
plt.title("Distribution of Modal Prices (Matplotlib hist)")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# --- Box plot by crop - seaborn ---
plt.figure(figsize=(10, 5))
sns.boxplot(x='Commodity', y='Modal_Price', data=df_selected)
plt.title("Price Distribution by Crop")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Scatter plot - seaborn ---
plt.figure(figsize=(10, 5))
sns.scatterplot(x='Min_Price', y='Max_Price', hue='Commodity', data=df_selected)
plt.title("Min Price vs Max Price by Crop (Seaborn)")
plt.tight_layout()
plt.show()

# --- Average price bar chart ---
plt.figure(figsize=(10, 5))
df_selected.groupby('Commodity')['Modal_Price'].mean().plot(kind='bar', color='coral')
plt.title("Average Modal Price by Crop")
plt.xlabel("Commodity")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Count plot (records per crop) ---
plt.figure(figsize=(10, 5))
sns.countplot(x='Commodity', data=df_selected)
plt.title("Number of Records per Crop")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#------------------SUBPLOTS — 4 CHARTS IN ONE FIGURE--------------------------


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Crop Price Overview — All Four Charts", fontsize=14, fontweight='bold')

# Plot 1: Avg price bar
df_selected.groupby('Commodity')['Modal_Price'].mean().plot(
    kind='bar', ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title("Avg Modal Price by Crop")
axes[0, 0].set_xlabel("Crop")
axes[0, 0].set_ylabel("Price")
axes[0, 0].tick_params(axis='x', rotation=30)

# Plot 2: Histogram
axes[0, 1].hist(df_selected['Modal_Price'], bins=30, color='salmon', edgecolor='white')
axes[0, 1].set_title("Price Frequency Distribution")
axes[0, 1].set_xlabel("Price")
axes[0, 1].set_ylabel("Frequency")

# Plot 3: Scatter (Min vs Max)
axes[1, 0].scatter(df_selected['Min_Price'], df_selected['Max_Price'],
                   alpha=0.4, color='purple')
axes[1, 0].set_title("Min Price vs Max Price")
axes[1, 0].set_xlabel("Min Price")
axes[1, 0].set_ylabel("Max Price")

# Plot 4: KDE
sns.kdeplot(df_selected['Modal_Price'], fill=True, ax=axes[1, 1], color='green')
axes[1, 1].set_title("Density Distribution of Price")
axes[1, 1].set_xlabel("Price")

plt.tight_layout()
plt.show()

#-------------------------OBJECTIVE 1 — PRICE TREND OVER TIME------------------------


# Overall trend (line chart)
plt.figure(figsize=(12, 6))
df_selected.groupby('Arrival_Date')['Modal_Price'].mean().plot(linewidth=1.5)
plt.title("Overall Price Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

# Crop-wise trend
plt.figure(figsize=(12, 6))
df_selected.groupby(['Arrival_Date', 'Commodity'])['Modal_Price'].mean().unstack().plot()
plt.title("Crop-wise Price Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

# Monthly trend
df_selected['Month'] = df_selected['Arrival_Date'].dt.to_period('M').astype(str)
plt.figure(figsize=(12, 6))
df_selected.groupby('Month')['Modal_Price'].mean().plot(marker='o', linewidth=1.5)
plt.title("Monthly Price Trend")
plt.xlabel("Month")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Yearly trend (bar chart )
df_selected['Year'] = df_selected['Arrival_Date'].dt.year
plt.figure(figsize=(10, 5))
df_selected.groupby('Year')['Modal_Price'].mean().plot(kind='bar', color='darkorange')
plt.title("Year-Wise Average Price")
plt.xlabel("Year")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

#-----------------------OBJECTIVE 3 — MARKET COMPARISON-------------------------


top_markets = df_selected['Market'].value_counts().head(5).index
df_market   = df_selected[df_selected['Market'].isin(top_markets)]

# Average price per market
plt.figure(figsize=(12, 6))
df_market.groupby('Market')['Modal_Price'].mean().plot(kind='bar', color='teal')
plt.title("Top Markets by Average Modal Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Arrivals per market
plt.figure(figsize=(10, 5))
df_market['Market'].value_counts().plot(kind='bar', color='slateblue')
plt.title("Top Markets by Number of Arrivals")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Crop-wise market comparison 
plt.figure(figsize=(12, 6))
crop_market_pivot = df_market.groupby(
    ['Market', 'Commodity'])['Modal_Price'].mean().unstack()
crop_market_pivot.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title("Crop-wise Price by Market (Stacked)")
plt.xlabel("Market")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Horizontal bar chart 
plt.figure(figsize=(10, 5))
df_market.groupby('Market')['Modal_Price'].mean().plot(kind='barh', color='darkgreen')
plt.title("Average Price by Market (Horizontal)")
plt.xlabel("Price")
plt.tight_layout()
plt.show()

#-----------------OBJECTIVE 4 — PRICE VOLATILITY ANALYSIS---------------------


df_selected['Price_Spread'] = df_selected['Max_Price'] - df_selected['Min_Price']

# Price spread histogram
plt.figure(figsize=(10, 5))
plt.hist(df_selected['Price_Spread'], bins=30, facecolor='crimson', edgecolor='white')
plt.xlim(0, 1000)
plt.title("Price Spread Distribution")
plt.xlabel("Spread (Max - Min)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Price spread box plot by crop
plt.figure(figsize=(10, 5))
sns.boxplot(x='Commodity', y='Price_Spread', data=df_selected)
plt.title("Price Spread by Crop")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Heatmap using plt.imshow 
plt.figure(figsize=(6, 4))
heat_data = df_selected[['Min_Price', 'Max_Price', 'Modal_Price', 'Price_Spread']].corr().values
plt.imshow(heat_data, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(4), ['Min', 'Max', 'Modal', 'Spread'], rotation=45)
plt.yticks(range(4), ['Min', 'Max', 'Modal', 'Spread'])
plt.title("Price Variables Heatmap (imshow)")
plt.tight_layout()
plt.show()

#-------------------------------PAIRPLOT — SEABORN-----------------------------------


print("\nGenerating Pairplot (this may take a moment)...")
pair_df = df_selected[['Min_Price', 'Max_Price', 'Modal_Price', 'Commodity']].copy()
pair_df = pair_df.sample(min(500, len(pair_df)), random_state=42)  
sns.pairplot(pair_df, hue='Commodity')
plt.suptitle("Pairplot — Crop Price Variables", y=1.02)
plt.show()

#------------------PIE CHART & DONUT CHART----------------------------
# Pie chart — records per crop
crop_counts = df_selected['Commodity'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(crop_counts.values, labels=crop_counts.index,
        autopct='%1.1f%%', shadow=True)
plt.title("Share of Records by Crop")
plt.tight_layout()
plt.show()

# Donut chart — share of avg price per crop (Code 6)
avg_prices = df_selected.groupby('Commodity')['Modal_Price'].mean()
plt.figure(figsize=(8, 6))
plt.pie(avg_prices.values, labels=avg_prices.index,
        autopct='%1.1f%%', radius=1)
plt.pie([1], colors=['white'], radius=0.55)   
plt.title("Average Price Contribution by Crop (Donut)")
plt.tight_layout()
plt.show()

#---------------STACKED BAR — YEARLY CROP PRICE---------------------- 


yearly_crop = df_selected.groupby(
    ['Year', 'Commodity'])['Modal_Price'].mean().unstack().fillna(0)

yearly_crop.plot(kind='bar', stacked=True, figsize=(12, 6),
                 colormap='Set2')
plt.title("Yearly Average Price — Stacked by Crop")
plt.xlabel("Year")
plt.ylabel("Avg Modal Price")
plt.xticks(rotation=45)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


#-------------------PRICE PREDICTION — LINEAR REGRESSION--------------------------  


print("\n========== PRICE PREDICTION — LINEAR REGRESSION ==========")

df_ml = df_selected.copy()
df_ml['Date_Ordinal'] = df_ml['Arrival_Date'].map(lambda x: x.toordinal())

X = df_ml[['Date_Ordinal', 'Min_Price', 'Max_Price']]
y = df_ml['Modal_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Coefficients:", model.coef_)
print("Intercept:   ", round(model.intercept_, 4))

y_pred = model.predict(X_test)

r2  = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\nR² Score : {round(r2, 4)}")
print(f"MSE      : {round(mse, 4)}")

# Visualization 1: Actual vs Predicted scatter
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Modal Price")
plt.tight_layout()
plt.show()

# Visualization 2: Regression line on 100 samples 
plt.figure(figsize=(8, 5))
plt.plot(y_test.values[:100], label="Actual",    color='navy',     linewidth=1.5)
plt.plot(y_pred[:100],         label="Predicted", color='crimson',  linewidth=1.5, linestyle='--')
plt.legend()
plt.title("Actual vs Predicted — First 100 Test Samples")
plt.xlabel("Sample Index")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

# Visualization 3: Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.5, color='purple')
plt.axhline(y=0, color='red', linewidth=1.5)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.show()

# Seaborn regplot — Min_Price vs Modal_Price 
plt.figure(figsize=(8, 5))
sns.regplot(x='Min_Price', y='Modal_Price', data=df_selected.sample(500, random_state=1),
            line_kws={"color": "red"}, scatter_kws={"alpha": 0.3})
plt.title("Regression: Modal Price vs Min Price")
plt.xlabel("Min Price")
plt.ylabel("Modal Price")
plt.grid(True)
plt.tight_layout()
plt.show()


#-----------------------BEST MANDI RECOMMENDATION SYSTEM------------------------------


def recommend_best_mandi(data, crop_name, top_n=3):
    """
    Recommends best mandis for a crop using:
    - Average modal price   (55% weight — higher = more profit)
    - Price stability       (35% weight — lower std dev = more reliable)
    - Listing count         (10% weight — more data = more trustworthy)
    """
    crop_df = data[data['Commodity'] == crop_name].copy()

    if crop_df.empty:
        print(f"No data for {crop_name}")
        return None

    market_stats = crop_df.groupby('Market')['Modal_Price'].agg(
        Avg_Price='mean',
        Std_Dev='std',
        Listing_Count='count'
    ).reset_index()

    # Require minimum 5 listings for reliability
    market_stats = market_stats[market_stats['Listing_Count'] >= 5]

    def normalize(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-9)

    market_stats['Price_Score']     = normalize(market_stats['Avg_Price'])
    market_stats['Stability_Score'] = 1 - normalize(market_stats['Std_Dev'])
    market_stats['Count_Score']     = normalize(market_stats['Listing_Count'])

    market_stats['Final_Score'] = (
        0.55 * market_stats['Price_Score'] +
        0.35 * market_stats['Stability_Score'] +
        0.10 * market_stats['Count_Score']
    )

    top_markets = market_stats.sort_values(
        'Final_Score', ascending=False).head(top_n)

    print(f"\n{'='*55}")
    print(f"  Top {top_n} Recommended Mandis for {crop_name}")
    print(f"{'='*55}")
    print(top_markets[['Market', 'Avg_Price', 'Std_Dev',
                        'Listing_Count', 'Final_Score']].to_string(index=False))

    # Subplot: Score + Avg Price side by side 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Best Mandis for {crop_name}", fontsize=13, fontweight='bold')

    axes[0].barh(top_markets['Market'], top_markets['Final_Score'], color='steelblue')
    axes[0].set_title("Recommendation Score")
    axes[0].set_xlabel("Score (0–1)")
    axes[0].invert_yaxis()

    axes[1].barh(top_markets['Market'], top_markets['Avg_Price'], color='seagreen')
    axes[1].set_title("Average Modal Price (₹)")
    axes[1].set_xlabel("Price")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()

    return top_markets


# Run for all selected crops
for crop in selected_crops:
    recommend_best_mandi(df_selected, crop_name=crop, top_n=3)

