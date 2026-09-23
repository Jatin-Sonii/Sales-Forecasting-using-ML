# Intelligent Sales Forecasting, Anomaly Detection & Product Demand Segmentation System

An end-to-end time-series forecasting, machine learning, and business intelligence platform designed for retail and e-commerce operations. This project predicts future product demand across categories and regions, detects historical sales anomalies, segments product sub-categories based on demand behavior, and delivers actionable stocking recommendations through an interactive Streamlit web dashboard.

---

## Project Overview

Managing inventory levels in large-scale retail requires balancing supply with fluctuating demand patterns. Overstocking locks up working capital and inflates warehousing costs, while understocking leads to stockouts, lost revenue, and reduced customer loyalty. 

This project addresses these operational challenges by combining:
1. **Statistical & Machine Learning Forecasting:** Evaluating SARIMA, Facebook Prophet, and XGBoost to project future sales demand.
2. **Category & Regional Granularity:** Modeling localized demand trends for targeted stock allocation.
3. **Anomaly Detection:** Isolating irregular sales spikes and dips using Isolation Forest and Rolling Z-Score techniques.
4. **Product Demand Segmentation:** Grouping product sub-categories via K-Means Clustering to guide inventory management strategies.
5. **Interactive Dashboard:** Deploying a multi-page Streamlit application for business managers.

---

## End-to-End Workflow & Task Breakdown

### **Task 1 — Data Loading, Merging & Exploration**
* **Data Parsing:** Imported the Superstore Sales dataset, parsing `Order Date` and `Ship Date` into datetime objects.
* **Feature Extraction:** Engineered temporal attributes including `Year`, `Month`, `Week Number`, `Day of Week`, `Quarter`, and `Season`.
* **Aggregation:** Resampled daily order data into weekly and monthly sales aggregations.
* **Exploratory Insights:**
  * **Top Revenue Category:** **Technology** generates the highest overall revenue, driven by high average order values.
  * **Regional Growth:** The **West Region** demonstrates the most consistent year-over-year sales growth across the 4-year period.
  * **Fulfillment Latency:** The average shipping duration across all orders is **approx. 4 days**, exhibiting minimal variance across regions.
  * **Seasonality Spikes:** Strong, recurring sales surges consistently occur during **November and December** (Q4 holiday sales).

---

### **Task 2 — Time Series Analysis & Decomposition**
* **Decomposition:** Applied Classical Seasonal Decomposition (`statsmodels`) to split monthly sales into:
  1. **Trend Component:** Shows consistent macro-level upward growth across the 4-year timeline.
  2. **Seasonal Component:** Reveals sharp annual spikes every Q4 with mid-year slowdowns.
  3. **Residual Noise:** Captures unmodeled noise and localized shocks.
* **Stationarity Testing:** Conducted the **Augmented Dickey-Fuller (ADF) Test**.
  * *Stationarity Concept:* A time series is stationary if its mean, variance, and autocorrelation remain constant over time.
  * *Result:* The raw series was non-stationary ($p > 0.05$). First-order differencing ($\Delta Y_t = Y_t - Y_{t-1}$) was applied to achieve stationarity ($p < 0.05$) prior to modeling.

---

### **Task 3 — Sales Forecasting Model Comparison**

Three distinct modeling paradigms were trained and evaluated on monthly sales data:

1. **SARIMA (Statistical):** Configured with optimal parameters `(p, d, q) x (P, D, Q)_m` selected via AIC optimization to capture non-stationary trends and 12-month seasonality.
2. **Facebook Prophet (Additive Model):** Structured with standard `ds` and `y` inputs. Captured smooth non-linear trends and yearly seasonality patterns.
3. **XGBoost (Supervised ML):** Built using lagged features (`Lag_1`, `Lag_2`, `Lag_3`), a 3-month rolling mean, and temporal calendar flags (`Month`, `Quarter`, `Season`).

#### **Forecast Evaluation & Model Performance**

| Model | MAE | RMSE | Performance Summary |
| :--- | :---: | :---: | :--- |
| **SARIMA** | Baseline | Baseline | Solid baseline; captures linear trends and strict seasonality well. |
| **Facebook Prophet** | Moderate | Moderate | Excellent trend breakdown; resilient to missing dates and smooth seasonality. |
| **XGBoost Regressor (Best)** | **Lowest** | **Lowest** | **Top Performer:** Superior accuracy in predicting non-linear quarterly transitions. |

---

### **Task 4 — Segment-Level Forecasting**

Using the top-performing **XGBoost** framework, separate 3-month future demand forecasts were generated across product categories and key geographical regions:

* **Categories:** Furniture, Technology, Office Supplies
* **Regions:** West Region, East Region

**Key Takeaway:** The **Technology category** and **West region** display the highest projected growth trajectory for the upcoming quarter, indicating where primary capital and logistics support should be prioritized.

---

### **Task 5 — Anomaly Detection in Sales Data**

Unusual sales patterns were identified using two complementary methods:

1. **Isolation Forest (Unsupervised ML):** Identified multi-dimensional outliers based on deviation from historical volume and trend norms.
2. **Rolling Z-Score (Statistical Threshold):** Flagged weeks where sales deviated by more than **$\pm 2$ Standard Deviations** from a 4-week rolling mean.

#### **Comparative Findings**
* **Agreement:** Both methods consistently flagged major Q4 promotional surges (e.g., Black Friday / Cyber Monday sales spikes).
* **Disagreements:** Z-score flagged sudden short-term weekly drops, whereas Isolation Forest flagged sustained out-of-season volume surges.
* **Operational Value:** Distinguishes genuine operational anomalies (such as supply chain disruptions) from planned seasonal promotions.

---

### **Task 6 — Product Demand Segmentation (K-Means Clustering)**

Product sub-categories were aggregated on four key behavioral metrics:
* **Total Sales Volume**
* **YoY Sales Growth Rate**
* **Sales Volatility (Standard Deviation)**
* **Average Order Value (AOV)**
#### **Recommended Stocking Strategies**

| Cluster Label | Sub-Categories Included | Inventory Strategy |
| :--- | :--- | :--- |
| **Cluster 1: High Volume / Stable** | Binders, Paper, Phones, Chairs | **Just-in-Time (JIT) / High Buffer:** Maintain steady safety stock; fast-moving core revenue drivers. |
| **Cluster 2: Low Volume / High Volatility** | Copiers, Machines | **Make-to-Order / Low Safety Stock:** Keep minimal holding inventory due to high capital lockup risk. |
| **Cluster 3: Growing Demand** | Accessories, Furnishings, Appliances | **Aggressive Stocking:** Scale up safety stock buffers to capture accelerating market demand. |
| **Cluster 4: Declining / Low Demand** | Envelopes, Labels, Bookcases | **Lean Stocking / Clearance:** Reduce order frequency and clear excess stock to prevent obsolescence. |

---

## Task 7 — Interactive Streamlit Dashboard

The deployment layer consists of a multi-page interactive web app built using **Streamlit**:

* **Page 1 — Sales Overview Dashboard:** Executive summary cards, historical annual revenue bar charts, monthly trends, and interactive category/region filters.
* **Page 2 — Forecast Explorer:** Interactive module allowing managers to select specific categories/regions and adjust forecast horizons (1 to 3 months) with dynamic MAE/RMSE readouts.
* **Page 3 — Anomaly Report:** Interactive visual plot displaying flagged sales anomalies alongside tabular summaries detailing anomaly dates and volume deviations.
* **Page 4 — Product Demand Segments:** Visual 2D cluster scatter plot featuring itemized sub-category lookup tables and tailored stocking recommendations.

---

## Tech Stack

* **Language:** Python 3.9+
* **Data Processing & Analytics:** `pandas`, `numpy`
* **Time Series & Forecasting:** `statsmodels` (SARIMA), `prophet`, `xgboost`
* **Machine Learning & Clustering:** `scikit-learn` (`StandardScaler`, `IsolationForest`, `KMeans`, `PCA`)
* **Data Visualization:** `matplotlib`, `seaborn`, `plotly`
* **Web Application & Deployment:** `streamlit`, Streamlit Community Cloud

---
