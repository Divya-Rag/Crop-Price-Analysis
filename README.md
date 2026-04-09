# Crop Price Analysis & Best Mandi Recommendation System

 A data-driven system that helps farmers in India decide "where to sell their crops" for maximum profit - built using Python, Pandas, Seaborn, and Scikit-learn.

##  Problem Statement

Farmers in India often don't know which mandi (agricultural market) will give them the best price for their crop. They rely on word of mouth, habit, or middlemen who may not always act in their best interest.

This project solves that problem using historical market price data — analysing trends, detecting anomalies, and recommending the "top 3 mandis" per crop based on average price, price stability, and data reliability.


##  Objectives

1.Analyse price trends over time (monthly, yearly, crop-wise) 
2.Study price distribution and volatility across crops 
3.Compare markets and identify the highest-paying mandis 
4.Understand relationships between Min, Max, and Modal prices 
5.Predict future modal prices using Linear Regression 
6.Recommend the best mandi per crop using a scoring system 



##  Project Structure


crop-price-analysis

(i) crop_price_analysis.py     # Main analysis & recommendation code
(ii) Project Data.csv          # Dataset (mandi price records)
(iii) README.md                # Project documentation


##  Tech Stack

 Tool | Purpose 

(i) Python 3 - Core programming language 
(ii) Pandas - Data loading, cleaning, manipulation 
(iii) NumPy - Numerical operations, Z-score calculation 
(iv) Matplotlib - Charts — bar, line, histogram, pie, donut 
(v) Seaborn - Statistical plots — boxplot, heatmap, pairplot, regplot 
(vi)Scikit-learn - Linear Regression, train-test split, R² & MSE 


##  Crops Analysed

-  Wheat
-  Green Peas
-  Mustard
-  Gur (Jaggery)



##  Key Features

### 1. Data Cleaning
-> Replaced zero values with `NaN` in price columns
-> Filled missing values using **median imputation**
-> Verified data integrity before and after cleaning

### 2. Outlier Detection
-> "IQR Method" — Q1, Q3, lower & upper bounds calculated per column
-> "Z-Score Method" - flags values more than 3 standard deviations from the mean
-> Visualised using Seaborn box plots

### 3. Statistical Analysis
-> Covariance matrix across price variables
-> Correlation matrix with heatmap visualization
-> Price spread (Max - Min) distribution analysis

### 4. Visualizations (15+ charts)
-> Histograms, KDE plots, box plots, scatter plots
-> Pairplot across all price variables by crop
-> Donut chart, stacked bar chart, subplots dashboard (2×2)
-> Heatmaps using both 'sns.heatmap' and 'plt.imshow'
-> Monthly, yearly, and crop-wise trend line charts

### 5. Price Prediction - Linear Regression
-> Features: 'Date_Ordinal', 'Min_Price', 'Max_Price'
-> Target: 'Modal_Price'
-> 80/20 train-test split
-> Evaluated using "R² Score" and "MSE"
-> Visualised with actual vs predicted plots and residual plot

### 6.  Mandi Recommendation System
Scores each market per crop using a weighted formula:


Final Score = 0.55 × Price Score
            + 0.35 × Stability Score
            + 0.10 × Listing Count Score


-> "Price Score" - higher average modal price = better
-> "Stability Score" - lower standard deviation = more reliable
-> "Count Score" - more records = more trustworthy data

Outputs the "Top 3 mandis" per crop with a ranked bar chart.



## ▶️ How to Run

1. Clone the repository

git clone https://github.com/Divya-Rag/crop-price-analysis.git
cd crop-price-analysis


2. Install dependencies

pip install pandas numpy matplotlib seaborn scikit-learn


3. Run the script

python crop_price_analysis.py


** Make sure 'Project Data.csv' is in the same folder as the script.



##  Sample Output



=======================================================
  Top 3 Recommended Mandis for Wheat
=======================================================
       Market   Avg_Price    Std_Dev  Listing_Count  Final_Score
     Orai APMC 2432.547800  56.342629             50     0.810404
      Ait APMC 2458.921613 113.321688             93     0.807550
Madhogarh APMC 2420.057851 101.329684            121     0.693341




<img width="1400" height="500" alt="Best mandis for Wheat" src="https://github.com/user-attachments/assets/5a2282fd-6549-402c-9cf9-d5e8f6877418" />




##  Future Scope

-> Connect to "Agmarknet API" (Govt. of India) for live real-time prices
->  Build a "WhatsApp bot or mobile app" so farmers can query by crop name
->  Integrate "weather & seasonal data" to improve price predictions
->  Expand coverage to "more crops and more states"
->  Upgrade from Linear Regression to "Random Forest or XGBoost" for better accuracy
->  This project is a working prototype of what "eNAM (National Agriculture Market)" does at its core



##  About

Built as a data science project to apply and demonstrate skills in:
EDA , Data Cleaning , Outlier Detection , Statistical Analysis , Data Visualization , Machine Learning , Recommendation Systems

Open to feedback, suggestions, and collaborations!



## 📄 License

This project is open source and available under the [MIT License](LICENSE).
