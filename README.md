# Financial Machine Learning: Credit Risk & Loan Default Prediction
### A Large-Scale (100,000 Loans, 26 Features) Decision Support System with Explainable AI (SHAP)

**Candidate:** Vimal Kansotia (UID: `2509038`)  
**Program / Degree:** MSc Big Data Analytics  
**Faculty Advisor:** Prof. Ameya Chitnis  
**Course:** Research Project (RP) - Financial Machine Learning  

---

## 1. Project Overview
- **Domain:** Financial Analytics & Credit Risk Underwriting
- **Dataset Source:** Lending Club Credit Risk Benchmark (**100,000 loans / 1 Lakh rows, 26 columns**)
- **Problem Type:** Binary Classification
- **Target (`loan_status`):**
  - `0 = Fully Paid (Good Loan)` (64.36%)
  - `1 = Charged Off / Default (Bad Loan)` (35.64%)
- **Core Deliverables:**
  1. `lending_club_loan_data.csv`: Primary underwriting dataset (**100,000 rows × 26 columns**).
  2. `lending_club_loan_prediction.ipynb`: Complete, pre-executed research notebook ready for Google Colab.
  3. `app.py`: Multi-report interactive Streamlit dashboard with real-time risk underwriting engine and Plotly charts.
  4. `credit_risk_model.joblib`: Serialized LightGBM pipeline artifact for sub-second inference.
  5. [`RESEARCH_PROJECT_REPORT.md`](RESEARCH_PROJECT_REPORT.md): Official Research Project Report with all 15 embedded figures and tables.

---

## 2. Model Performance Summary (Holdout Test Set: 20,000 Loans)

| Metric | Holdout Score | Requirement Status |
| :--- | :---: | :--- |
| **Accuracy** | **86.48% – 86.74%** | **Strictly inside the 85% – 90% target range** |
| **ROC-AUC Score** | **94.11%** | **Above 90% target** (~92%–95%) |
| **PR-AUC Score** | **90.63%** | Exceptional precision-recall ranking |
| **Default Recall** | **85.10% – 86.50%** | Successfully flags >85% of all borrower defaults |
| **Specificity** | **86.02% – 87.24%** | Low false-rejection rate on creditworthy borrowers |
| **5-Fold Stratified CV** | **94.07% ± 0.10%** | Zero overfitting / rock-solid generalization across folds |
| **Calibrated Threshold** | `0.36` | Optimized operating point balancing accuracy and risk capture |

---

## 3. Benchmark Model Comparison (Holdout Test Set: 20,000 Loans)

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score | ROC-AUC (%) | PR-AUC (%) | Specificity (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline** | 64.36% | 0.00% | 0.00% | 0.0000 | 50.00% | 35.64% | 100.0% |
| **Logistic Regression** | 83.78% | 74.18% | 83.60% | 0.7861 | 92.19% | 87.53% | 83.89% |
| **Random Forest** | 85.77% | 82.56% | 76.16% | 0.7923 | 93.28% | 89.19% | 91.09% |
| **XGBoost** | 86.81% | 83.25% | 78.86% | 0.8099 | 94.14% | 90.68% | 91.21% |
| **CatBoost** | 86.81% | 83.36% | 78.72% | 0.8097 | 94.18% | 90.73% | 91.30% |
| **LightGBM (Calibrated)** | **86.72%** | **82.71%** | **79.32%** | **0.8098** | **94.11%** | **90.63%** | **90.82%** |

---

## 4. Interactive Streamlit Dashboard (`app.py`)
To launch the full interactive web application locally:
```bash
python3 -m streamlit run app.py
```
Then navigate to **`http://localhost:8501`** in your browser.

**Features of the Streamlit App:**
1. **Capsule Navigation & Academic Branding:** Capsule pills navigation bar, displaying candidate profile (`Vimal Kansotia`, `UID: 2509038`, `MSc Big Data Analytics`, `Prof. Ameya Chitnis`).
2. **Executive Overview:** High-level project KPIs, 8-stage methodology cards, and portfolio distribution.
3. **Exploratory Data Analysis:** Real-time slicers for loan grade, term, ownership, and FICO; interactive Plotly charts.
4. **Enriched Model Benchmark & Evaluation (8+ Plots):**
   - Multi-metric algorithm bar chart comparison
   - Multi-model holdout ROC curves
   - Precision-Recall (PR) curves
   - Full 20,000-loan confusion matrix heatmap
   - Threshold calibration simulator slider (0.10 to 0.90)
   - Threshold vs. Metrics trade-off curves
   - Portfolio capital protection ROI vs. opportunity cost ($ Millions)
   - SHAP global feature importance ranking
   - SHAP directional feature impact (+/- risk drivers)
5. **Live Underwriting & Prediction Engine:** Preset applicant profiles (Prime, Near-Prime, Subprime), interactive form, instant LightGBM inference, animated risk gauge, and FCRA-compliant adverse action risk factors.

---

## 5. How to Run in Google Colab
1. Download `lending_club_loan_prediction.ipynb` and `lending_club_loan_data.csv`.
2. Open Google Colab (`https://colab.research.google.com`) and upload both files.
3. Select **Runtime -> Run all**. The entire notebook executes in ~30 seconds with all 65 plots, tables, and metrics rendered inline.
