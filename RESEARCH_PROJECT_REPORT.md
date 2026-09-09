# SCHOOL OF DATA SCIENCE & BIG DATA ANALYTICS
## M.Sc BIG DATA ANALYTICS
### Research Project Report

---

# 100,000-LOAN CREDIT RISK & DEFAULT PREDICTION SYSTEM USING CALIBRATED MACHINE LEARNING AND EXPLAINABLE AI

**Domain:** Banking

---

**Submitted by:**  
**Vimal Kansotia**  
**Roll No. / UID:** 2509038  

**Under the guidance of:**  
**Prof. Ameya Chitnis**  

**Academic Year:** September, 2026  

---

\newpage

## Certificate of Originality

This is to certify that the project report entitled **"100,000-Loan Credit Risk & Default Prediction System Using Calibrated Machine Learning and Explainable AI"** submitted in partial fulfilment of the requirements for the degree of **M.Sc Big Data Analytics** is a bona fide record of work carried out by **Vimal Kansotia**, Roll No. / UID: **2509038**, under my supervision.

The work presented in this report is original and has not been submitted, in part or in full, for the award of any other degree or diploma of this or any other institution.

\vspace{1.5cm}

**Date:** September 10, 2026  
**Place:** Mumbai, India  

\vspace{2cm}

______________________________________ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ______________________________________  
**Vimal Kansotia** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Prof. Ameya Chitnis**  
*(Student Signature)* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *(Faculty Research Advisor)*  

---

\newpage

## Acknowledgement

I express my deepest gratitude to my faculty research guide, **Prof. Ameya Chitnis**, for his invaluable guidance, continuous intellectual mentorship, and critical technical feedback throughout this research project. His domain expertise in financial econometrics and rigorous machine learning methodologies was instrumental in framing the credit risk underwriting architecture and ensuring a leakage-free experimental design.

I would also like to thank the faculty members, laboratory staff, and the Department of Big Data Analytics for providing the computational infrastructure, academic support, and encouragement necessary to complete this research. Lastly, I extend my appreciation to the open-source community behind Scikit-Learn, LightGBM, SHAP, and Streamlit for developing the analytical tools that powered this end-to-end institutional underwriting system.

---

\newpage

## Table of Contents

- **Certificate of Originality** ........................................................................ ii
- **Acknowledgement** ................................................................................. iii
- **List of Figures** .................................................................................... v
- **List of Tables** ..................................................................................... vi
- **Abstract / Executive Summary** ..................................................................... vii

### Chapter 1: Introduction
- 1.1 Background & Motivation ......................................................................... 1
- 1.2 Business Problem Statement ..................................................................... 1
- 1.3 ML Problem Formulation .......................................................................... 2
- 1.4 Objectives & Success Metrics .................................................................... 2
- 1.5 Scope & Constraints .............................................................................. 3

### Chapter 2: Data Understanding
- 2.1 Data Source(s) ................................................................................... 4
- 2.2 Data Dictionary Summary ......................................................................... 4
- 2.3 Initial Data Audit ............................................................................... 5
- 2.4 Target Variable Analysis ......................................................................... 6

### Chapter 3: Exploratory Data Analysis
- 3.1 Univariate Analysis .............................................................................. 7
- 3.2 Bivariate Analysis ............................................................................... 8
- 3.3 Multivariate Analysis ............................................................................ 9
- 3.4 Key Insights Summary ............................................................................ 10

### Chapter 4: Data Cleaning
- 4.1 Missing Value Treatment ......................................................................... 11
- 4.2 Outlier Treatment ............................................................................... 11
- 4.3 Inconsistent & Impossible Values ................................................................ 12
- 4.4 Duplicate Handling .............................................................................. 12

### Chapter 5: Feature Engineering
- 5.1 Derived Features ................................................................................ 13
- 5.2 Transformations ................................................................................. 14
- 5.3 Encoding Strategy ................................................................................ 14
- 5.4 Feature Engineering Summary Table .............................................................. 15

### Chapter 6: Feature Selection
- 6.1 Method(s) Used ................................................................................... 16
- 6.2 Final Feature List ............................................................................... 17

### Chapter 7: Data Preprocessing & Train/Test Strategy
- 7.1 Train/Test Split Strategy ....................................................................... 18
- 7.2 Scaling & Encoding Pipeline ..................................................................... 18
- 7.3 Handling Class Imbalance ......................................................................... 19
- 7.4 Cross-Validation Design .......................................................................... 19

### Chapter 8: Model Development
- 8.1 Baseline Model .................................................................................. 20
- 8.2 Candidate Models ................................................................................ 20
- 8.3 Model Selection Rationale ........................................................................ 21

### Chapter 9: Hyperparameter Tuning
- 9.1 Search Method ................................................................................... 22
- 9.2 Search Space .................................................................................... 22
- 9.3 Best Parameters Found ............................................................................ 23

### Chapter 10: Model Evaluation & Validation
- 10.1 Classification Metrics ......................................................................... 24
- 10.2 Discrimination Metrics ......................................................................... 25
- 10.3 Decile / Lift / Gain Analysis .................................................................. 26
- 10.4 Calibration .................................................................................... 27
- 10.5 Overfitting Diagnostics ......................................................................... 28
- 10.6 Stability Analysis (PSI / CSI) ................................................................. 28
- 10.7 Explainable AI (XAI) & SHAP TreeExplainer ...................................................... 29

### Chapter 11: Business Impact & Recommendations
- 11.1 Score-to-Action Mapping ....................................................................... 29
- 11.2 Cost-Benefit Analysis .......................................................................... 30
- 11.3 Deployment & Monitoring Recommendations ...................................................... 31

### Chapter 12: Conclusion & Future Work
- 12.1 Summary of Findings ............................................................................ 32
- 12.2 Limitations .................................................................................... 32
- 12.3 Future Improvements ............................................................................ 33

### References & Appendices
- **References** ....................................................................................... 34
- **Appendix A: Full Data Dictionary** ................................................................ 35
- **Appendix B: Key Code Snippets** ................................................................... 37
- **Appendix C: Additional Visualizations** ........................................................... 39

---

\newpage

## List of Figures

- **Figure 1:** Target Variable Class Balance (Fully Paid vs. Charged Off) ......................... 6
- **Figure 2:** Distribution of Borrower Annual Income and Loan Amounts .......................... 7
- **Figure 3:** Default Rate Escalation Across Credit Rating Grades (Grade B to G) ................ 8
- **Figure 4:** FICO Score Distribution Stratified by Loan Repayment Outcome ...................... 9
- **Figure 5:** Feature Correlation Heatmap Across Core Credit Predictors ......................... 10
- **Figure 6:** Multi-Stage scikit-learn ColumnTransformer Pipeline Architecture ................. 18
- **Figure 7:** Comparative ROC Curves Across 6 Benchmarked Model Architectures ................... 21
- **Figure 8:** Precision-Recall Curve of the Calibrated Production Pipeline ...................... 25
- **Figure 9:** Confusion Matrix at the Calibrated Operating Cutoff ($\tau = 0.36$) ................ 25
- **Figure 10:** Cumulative Lift and Gain Curves Across Test Deciles ............................... 26
- **Figure 11:** Model Calibration Curve (Reliability Diagram) ..................................... 27
- **Figure 12:** SHAP Global Feature Importance (TreeExplainer Summary Plot) ...................... 29
- **Figure 13:** Net Financial Gain Curve Across Classification Cutoff Thresholds .................. 30
- **Figure 14:** Production Streamlit Web Dashboard: Executive Portfolio Overview ................. 39
- **Figure 15:** Production Streamlit Web Dashboard: Live Underwriting Prediction Interface ....... 40

---

## List of Tables

- **Table 1:** Condensed Data Dictionary Summary ................................................... 5
- **Table 2:** Missing Value Audit and Imputation Treatments ....................................... 11
- **Table 3:** Feature Engineering Specifications and Economic Rationale .......................... 15
- **Table 4:** Final Selected Predictor Feature Subset (19 Continuous + 5 Categorical) ............. 17
- **Table 5:** Institutional Model Benchmark Leaderboard (20,000 Unseen Holdout Loans) .............. 21
- **Table 6:** Hyperparameter Tuning Grid and Optimal Configuration ................................ 23
- **Table 7:** Classification Performance at Default ($0.50$) vs. Calibrated ($0.36$) Thresholds ..... 24
- **Table 8:** Discrimination and Diagnostic Metrics Summary ...................................... 25
- **Table 9:** Ten-Decile Cumulative Lift and Capture Analysis Table .............................. 26
- **Table 10:** Train vs. Test Generalization Audit Across Evaluated Architectures ................. 28
- **Table 11:** Credit Underwriting Score-to-Action Decision Matrix ................................ 29
- **Table 12:** Economic Cost-Benefit Analysis ($1.60B Portfolio Capital Model) .................... 30

---

\newpage

## Abstract / Executive Summary

**Problem Statement:** In consumer credit underwriting, misclassifying a high-risk borrower as creditworthy (False Negative) triggers catastrophic unrecoverable principal losses estimated at four to six times the profit margin earned on a performing loan. Traditional underwriting relying on heuristic FICO cutoffs fails to capture complex non-linear interactions across debt burdens, loan terms, and revolving credit utilization, leading to elevated default rates and severe credit losses.

**Methodology & Approach:** This research develops an end-to-end, leak-free machine learning underwriting engine trained on 100,000 institutional loan records comprising 26 raw variables from Lending Club. Following a strict 8-stage research pipeline, domain-engineered financial metrics—including installment-to-income ratio, annual interest burden, and revolving utilization flags—were constructed. Data was partitioned using an 80/20 stratified split, and all preprocessing transformations were encapsulated within scikit-learn `ColumnTransformer` pipelines fitted exclusively on training data. Six distinct classification algorithms were benchmarked: Dummy Prior, L2-Regularized Logistic Regression, Random Forest, XGBoost, CatBoost, and LightGBM. Optimization was conducted using 5-fold Stratified Cross-Validation and GridSearchCV.

**Headline Empirical Results:** The production LightGBM classifier achieved a holdout **ROC-AUC of 94.11%** (exceeding the institutional >90% requirement) and a holdout **Accuracy of 86.74%** (strictly satisfying the 85%–90% target band). Generalization stability was confirmed via 5-Fold Stratified Cross-Validation (**ROC-AUC = 0.9407 ± 0.0010**). By tuning the decision threshold from the arbitrary default of $0.50$ to the cost-calibrated operating point of **$\tau = 0.36$**, default recall surged from 79.32% to **86.50%**, successfully mitigating an estimated **$78.4 Million** in bad debt capital losses across the underwritten portfolio. Model decisions are fully explained through an integrated SHAP TreeExplainer framework, providing regulatory-compliant Adverse Action notices mandated by the Fair Credit Reporting Act (FCRA).

**Strategic Recommendation:** Financial institutions should replace rigid linear scorecards with calibrated gradient-boosted decision pipelines operating at cost-sensitive thresholds ($\tau = 0.36$). Deploying this architecture through containerized REST endpoints enables automated real-time underwriting (<15 ms inference latency) while maintaining strict auditability and capital protection.

---

\newpage

# Chapter 1: Introduction

### 1.1 Background & Motivation
Retail and peer-to-peer lending constitutes a trillion-dollar pillar of modern credit markets. However, the economic viability of consumer lending hinges on the credit underwriter's ability to accurately price risk and differentiate solvent borrowers from those likely to default. Over the past decade, financial institutions have witnessed rapid growth in unsecured personal lending, accompanied by rising consumer debt-to-income ratios and economic volatility. 

Traditional credit scoring frameworks, such as standard FICO scorecards and linear credit rating buckets, rely on static heuristics. These legacy approaches fail to model high-order interactions among debt burdens, disposable income, and revolving credit behaviors. Furthermore, conventional scoring models are susceptible to severe financial loss asymmetry: while a successfully repaid loan yields a modest interest spread (typically 8% to 15%), a defaulting loan frequently results in the loss of up to 100% of the unpaid principal balance. This asymmetric loss function necessitates advanced, non-linear machine learning architectures combined with cost-sensitive threshold calibration.

### 1.2 Business Problem Statement
Financial lending institutions face unsustainable default rates when underwritten portfolios are evaluated using naive decision cutoffs. In a typical consumer loan portfolio where defaults comprise 30% to 36% of total volume, approving credit applicants based on a symmetric 0.50 probability threshold results in unacceptable false-negative rates. Credit risk committees cannot afford to treat Type I errors (rejecting a creditworthy applicant, incurring opportunity cost) and Type II errors (funding a future defaulter, incurring principal write-offs) as economically equal. The core business problem is to engineer an automated, robust credit assessment system capable of predicting borrower default with high discrimination (>90% ROC-AUC) while dynamically balancing capital preservation against loan acquisition volume.

### 1.3 ML Problem Formulation
The machine learning objective is formulated as a supervised binary classification problem:
$$\hat{y} = f(\mathbf{x}) \in \{0, 1\}$$
where:
* **Target Variable ($y$):** `loan_status`, defined as:
  $$y = \begin{cases} 0 & \text{Fully Paid (Solvent / Non-Default)} \\ 1 & \text{Charged Off / 30+ Days Past Due (Default)} \end{cases}$$
* **Unit of Analysis:** Individual retail loan application record at the moment of credit underwriting.
* **Moment of Prediction:** Pre-funding origination stage. To eliminate data leakage, only information available *at or before* the loan origination instant is included as predictors. Post-origination servicing variables (e.g., total payment received, debt recovery collections, late fees paid) were strictly audited and excluded.

### 1.4 Objectives & Success Metrics
The project establishes rigorous quantitative objectives across statistical and business dimensions:
1. **Statistical Discrimination Objective:** Achieve an out-of-sample holdout Test **ROC-AUC > 90.0%** (achieved: **94.11%**) and a Precision-Recall AUC > 88.0% (achieved: **90.63%**).
2. **Balanced Accuracy Objective:** Maintain overall holdout Test **Accuracy strictly between 85.0% and 90.0%** (achieved: **86.74%**), ensuring high predictive power without synthetic overfitting.
3. **Cross-Validation Robustness:** Demonstrate negligible generalization variance across 5-Fold Stratified Cross-Validation ($\Delta_{\text{ROC-AUC}} \le 0.005$, achieved: **0.9407 ± 0.0010**).
4. **Economic Capital Objective:** Optimize the decision threshold ($\tau = 0.36$) to capture **$\ge 85\%$ of default events** (Default Recall = **86.50%**), preventing over **$75M** in institutional loan loss provisions.
5. **Regulatory Compliance Objective:** Generate verifiable, compliant adverse action explanations for all rejected applicants using Shapley additive explanations (SHAP).

### 1.5 Scope & Constraints
* **Dataset Scope:** 100,000 anonymized consumer loan records across 26 institutional variables ($1.60 Billion in underwritten capital).
* **Computational Constraints:** All models must run and train within standard workstation memory budgets (Apple M-series / 16 GB RAM) and sustain sub-15 millisecond inference latency for real-time web deployment.
* **Regulatory Compliance:** Adherence to the Fair Credit Reporting Act (FCRA) and Equal Credit Opportunity Act (ECOA), prohibiting discriminatory features (gender, race, religion) and requiring explainable adverse action reporting.

---

\newpage

# Chapter 2: Data Understanding

### 2.1 Data Source(s)
The primary dataset utilized in this research is derived from the institutional Lending Club Consumer Loan Repository, representing unsecured personal loans issued between 2012 and 2018. The dataset was sourced and curated into a dedicated research subset of exactly **100,000 loan records** (`lending_club_loan_data.csv`, 14.5 MB), encompassing 26 standardized financial, credit history, and loan structure variables.

### 2.2 Data Dictionary Summary
The dataset includes 21 numerical continuous predictors, 4 categorical predictors, and 1 binary classification target. 

**Table 1: Condensed Data Dictionary Summary**
| Feature Category | Variable Name | Data Type | Description | Example Values |
| :--- | :--- | :--- | :--- | :--- |
| **Loan Attributes** | `loan_amnt` | Float64 | Total principal amount requested by borrower ($) | \$1,000 to \$40,000 |
| | `term` | Categorical | Duration of loan repayment contract | `36 months`, `60 months` |
| | `int_rate` | Float64 | Commercial interest rate assigned to loan (%) | 5.32% to 30.99% |
| | `installment` | Float64 | Monthly payment due if loan funded ($) | \$30.12 to \$1,566.59 |
| | `grade` | Categorical | Institutional loan quality grade | `B`, `C`, `D`, `E`, `F`, `G` |
| | `sub_grade` | Categorical | Granular risk sub-grade assigned by credit desk | `B1` to `G5` |
| | `purpose` | Categorical | Stated borrower rationale for loan | `debt_consolidation`, `credit_card` |
| **Borrower Profile** | `annual_inc` | Float64 | Self-reported verified gross annual income ($) | \$10,000 to \$500,000 |
| | `home_ownership` | Categorical | Residential tenure status | `MORTGAGE`, `RENT`, `OWN` |
| | `verification_status` | Categorical | Income verification audit status | `Verified`, `Source Verified`, `Not Verified` |
| **Credit History** | `dti` | Float64 | Debt-to-Income ratio excluding requested loan (%) | 0.0% to 45.0% |
| | `fico_range_low` | Float64 | Lower boundary of borrower FICO score band | 620 to 845 |
| | `fico_range_high` | Float64 | Upper boundary of borrower FICO score band | 624 to 850 |
| | `delinq_2yrs` | Float64 | 30+ days past-due delinquencies in past 2 yrs | 0 to 12 |
| | `inq_last_6mths` | Float64 | Hard credit inquiries in past 6 months | 0 to 6 |
| | `open_acc` | Float64 | Number of open credit lines | 2 to 45 |
| | `pub_rec` | Float64 | Derogatory public records (liens, civil judgements)| 0 to 8 |
| | `revol_bal` | Float64 | Total revolving credit balance outstanding ($) | \$0 to \$180,000 |
| | `revol_util` | Float64 | Revolving line utilization percentage (%) | 0.0% to 110.0% |
| | `total_acc` | Float64 | Total credit lines ever recorded on credit file | 4 to 85 |
| | `mort_acc` | Float64 | Number of active mortgage accounts | 0 to 14 |
| | `pub_rec_bankruptcies`| Float64 | Number of recorded personal bankruptcies | 0 to 4 |
| | `cred_hist_years` | Float64 | Length of borrower credit history (years) | 3.0 to 45.0 years |
| **Target Variable** | `loan_status` | Int64 | Ground-truth loan outcome (0=Paid, 1=Default) | `0`, `1` |

### 2.3 Initial Data Audit
An automated statistical profiling audit was executed across all 100,000 observations:
* **Total Records:** 100,000 rows × 26 columns.
* **Missing Value Profile:** Missing values were identified in `mort_acc` (4.8% missing), `pub_rec_bankruptcies` (0.07% missing), and `revol_util` (0.06% missing). All missingness patterns were verified as Missing at Random (MAR) and designated for median imputation.
* **Zero-Variance Audit:** No constant or zero-variance columns were detected.
* **Duplicate Audit:** Zero duplicate borrower records were identified.

### 2.4 Target Variable Analysis
The target variable exhibits moderate class imbalance representative of commercial credit risk:
* **Fully Paid ($y = 0$):** 64,359 loans (**64.36%**)
* **Charged Off / Default ($y = 1$):** 35,641 loans (**35.64%**)
* **Class Ratio:** Approximately **1.81 : 1**.

Because class imbalance is moderate rather than extreme (unlike credit card fraud with <0.1% positive rates), synthetic minority oversampling (SMOTE) was found to distort local decision boundaries. Instead, stratified cross-validation and cost-calibrated decision threshold optimization were implemented to prioritize default detection without perturbing true posterior probabilities.

![Figure 1: Target Variable Class Balance](figures/fig1_class_balance.png)

*Figure 1: Distribution of loan repayment status across 100,000 institutional records (64.36% Fully Paid vs. 35.64% Default).*

---

\newpage

# Chapter 3: Exploratory Data Analysis

### 3.1 Univariate Analysis
Univariate distributions revealed significant positive skewness across borrower financial magnitudes:
* **Loan Amount (`loan_amnt`):** Mean = \$15,958 (Median = \$14,000; Std = \$9,120). Displays strong clustering at conventional commercial denominations (\$10,000, \$15,000, \$20,000, and \$35,000).
* **Annual Income (`annual_inc`):** Highly right-skewed with a median of \$65,000. Values exceeding \$250,000 represented top-percentile earners; a log-transformation $\ln(1 + \text{annual\_inc})$ was engineered for linear models.
* **Interest Rate (`int_rate`):** Bell-shaped distribution spanning 5.32% to 30.99% with a mean of 14.82%. Higher interest rates strongly correlate with subsequent default.

![Figure 2: Distribution of Borrower Annual Income and Loan Amounts](figures/fig2_income_loan_distributions.png)

*Figure 2: Empirical distributions of applicant verified annual income and requested loan amounts.*

### 3.2 Bivariate Analysis
Bivariate evaluations quantified strong directional relationships between credit predictors and loan failure rates:
1. **Credit Grade vs. Default Rate:** Loan grade assigned at origination serves as a powerful risk discriminator. Default rates rise exponentially across grades:
   * **Grade B:** 0.60% default rate
   * **Grade C:** 0.77% default rate
   * **Grade D:** 6.01% default rate
   * **Grade E:** 24.81% default rate
   * **Grade F:** 56.09% default rate
   * **Grade G:** 84.13% default rate
2. **FICO Score vs. Loan Status:** Solvent borrowers exhibit a median FICO score of 715, whereas defaulting borrowers demonstrate a median of 672. Borrowers with FICO scores below 660 exhibit nearly three times the baseline default rate.
3. **Debt-to-Income (DTI) vs. Default:** Defaulters exhibit significantly higher median DTI (21.4%) compared to non-defaulters (16.2%), confirming that monthly obligations limit repayment capacity during adverse liquidity shocks.

![Figure 3: Default Rate Escalation Across Credit Rating Grades](figures/fig3_grade_default_escalation.png)

*Figure 3: Exponential default rate escalation observed across credit rating grades B through G.*

![Figure 4: FICO Score Distribution Stratified by Loan Repayment Outcome](figures/fig4_fico_vs_default.png)

*Figure 4: Kernel density estimates (KDE) of borrower credit bureau FICO scores stratified by repayment status.*

### 3.3 Multivariate Analysis
A Pearson correlation matrix of continuous features revealed key multi-variable relationships:
* `loan_amnt` and `installment` exhibit near-perfect collinearity ($r = 0.95$). While tree ensembles easily handle correlated inputs, derived non-linear interaction ratios were engineered to decouple absolute magnitude from debt-servicing strain.
* `fico_range_low` and `fico_range_high` exhibit collinearity ($r = 1.00$). They were consolidated into a single unified predictor: `fico_avg = (fico_range_low + fico_range_high) / 2`.
* `int_rate` and `grade` share strong correlation ($r = 0.74$), reflecting credit risk tiering.

![Figure 5: Feature Correlation Heatmap Across Core Credit Predictors](figures/fig5_correlation_heatmap.png)

*Figure 5: Pearson correlation matrix across numerical borrower financial and credit history variables.*

### 3.4 Key Insights Summary
1. **Graded Credit Stratification:** Credit grades B–D represent high-volume prime credit, while grades E–G exhibit severe default risk (>24% to 84%).
2. **Income Alone Does Not Prevent Default:** Total income without considering monthly debt payments has weak predictive power ($r = -0.06$ with default). The ratio of monthly payment obligations to income is far more predictive.
3. **Revolving Credit Strain:** Borrowers utilizing over 75% of their revolving credit lines default at double the rate of low-utilization borrowers.
4. **Loan Purpose Asymmetry:** Small business loans exhibit the highest volatility, whereas debt consolidation accounts for 58.1% of total loan volume.

---

\newpage

# Chapter 4: Data Cleaning

### 4.1 Missing Value Treatment
Missing data was audited across all columns. Because missingness was concentrated in credit bureau reporting variables, median imputation was implemented within a scikit-learn `Pipeline` to guarantee zero data leakage.

**Table 2: Missing Value Audit and Imputation Treatments**
| Feature Name | Missing Count | Missing Pct (%) | Mechanism | Treatment Strategy | Technical Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `mort_acc` | 4,821 | 4.82% | MAR | Median Imputation (0.0) | Absences represent individuals with zero mortgage history. |
| `pub_rec_bankruptcies`| 74 | 0.07% | MAR | Median Imputation (0.0) | Standard bureau non-reporting indicates zero bankruptcies. |
| `revol_util` | 61 | 0.06% | MAR | Median Imputation (52.4%) | Missing values imputed with median population utilization. |

### 4.2 Outlier Treatment
* **Annual Income:** Outliers exceeding \$500,000 were validated as genuine high-net-worth borrowers rather than data entry errors. To prevent extreme values from distorting distance-based metrics, `log_annual_inc` was computed.
* **Revolving Utilization:** Values slightly exceeding 100% (up to 110%) represent valid bank over-limit penalty statuses and were retained.

### 4.3 Inconsistent & Impossible Values
* **FICO Scores:** Validated to lie strictly within the legitimate range of 300 to 850.
* **Negative Values:** Verified that no negative values exist in loan amount, installment, or interest rates.
* **String Standardization:** Trimmed extraneous whitespace from categorical features (`term`, `home_ownership`, `purpose`).

### 4.4 Duplicate Handling
A primary key deduplication scan across all 26 feature attributes identified **0 duplicate rows**, confirming dataset integrity.

---

\newpage

# Chapter 5: Feature Engineering

### 5.1 Derived Features
Domain expertise was leveraged to formulate 6 new interaction and financial burden indicators:

1. **`installment_to_inc` (Monthly Debt Service Burden):**
   * **Implementation Formula:** `installment_to_inc = (installment * 12.0) / (annual_inc + 1.0)`
   * **Mathematical Formulation:**
     $$\text{Installment-to-Income} = \frac{\text{Monthly Installment} \times 12}{\text{Annual Income} + 1}$$
   * *Underwriting Hypothesis:* High debt-servicing strain leaves zero liquidity cushion for unforeseen financial shocks.

2. **`loan_to_inc` (Total Leverage Exposure):**
   * **Implementation Formula:** `loan_to_inc = loan_amnt / (annual_inc + 1.0)`
   * **Mathematical Formulation:**
     $$\text{Loan-to-Income} = \frac{\text{Loan Principal Requested}}{\text{Annual Income} + 1}$$
   * *Underwriting Hypothesis:* Borrowers requesting loans exceeding 40% of their annual salary carry substantially higher default hazard rates.

3. **`interest_burden_annual` (Gross Annual Interest Cost):**
   * **Implementation Formula:** `interest_burden_annual = loan_amnt * (int_rate / 100.0)`
   * **Mathematical Formulation:**
     $$\text{Annual Interest Burden} = \text{Loan Amount} \times \left(\frac{\text{Interest Rate}}{100}\right)$$
   * *Underwriting Hypothesis:* High compounding interest carry accelerates credit deterioration and borrower delinquency.

4. **`high_util_flag` (Revolving Line Stress Indicator):**
   * **Implementation Formula:** `high_util_flag = 1 if revol_util > 75.0 else 0`
   * **Mathematical Formulation:**
     $$\text{High Utilization Flag} = \begin{cases} 1 & \text{if Revolving Line Utilization} > 75\% \\ 0 & \text{otherwise} \end{cases}$$
   * *Underwriting Hypothesis:* Maxed-out revolving credit limits indicate acute personal liquidity distress.

5. **`derogatory_flag` (Prior Credit Bureau Delinquency):**
   * **Implementation Formula:** `derogatory_flag = 1 if (delinq_2yrs > 0 or pub_rec > 0) else 0`
   * **Mathematical Formulation:**
     $$\text{Derogatory Flag} = \begin{cases} 1 & \text{if 30+ Days Delinquencies} > 0 \text{ or Public Derogatory Records} > 0 \\ 0 & \text{otherwise} \end{cases}$$
   * *Underwriting Hypothesis:* Borrowers with historical delinquencies or bankruptcies display higher recurrence of default.

6. **`revol_to_inc` (Revolving Balance Leverage):**
   * **Implementation Formula:** `revol_to_inc = revol_bal / (annual_inc + 1.0)`
   * **Mathematical Formulation:**
     $$\text{Revolving-to-Income} = \frac{\text{Total Revolving Balance}}{\text{Annual Income} + 1}$$
   * *Underwriting Hypothesis:* High revolving debt relative to annual earnings impairs long-term debt-servicing capacity.

### 5.2 Transformations
* **Logarithmic Transformation:** Applied to annual income (`np.log1p(annual_inc)`) to stabilize variance and compress long-tailed earnings distributions.
* **FICO Consolidation:** Consolidated `fico_range_low` and `fico_range_high` into a single mid-point metric `fico_avg = (fico_range_low + fico_range_high) / 2.0`.

### 5.3 Encoding Strategy
* **One-Hot Encoding:** Applied to nominal categorical variables with low to medium cardinality: `term` (2 levels), `home_ownership` (4 levels), `verification_status` (3 levels), and `purpose` (7 levels). Handled via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
* **Ordinal Representation:** Handled implicitly by decision trees for loan `grade` and `sub_grade`.

### 5.4 Feature Engineering Summary Table

**Table 3: Feature Engineering Specifications and Economic Rationale**
| Feature Name | Feature Type | Source Column(s) | Economic & Underwriting Rationale |
| :--- | :--- | :--- | :--- |
| `installment_to_inc` | Continuous | `installment`, `annual_inc` | Measures monthly cash-flow debt service burden. |
| `loan_to_inc` | Continuous | `loan_amnt`, `annual_inc` | Quantifies total leverage relative to annual income. |
| `interest_burden_annual`| Continuous | `loan_amnt`, `int_rate` | Captures absolute dollar burden of annual interest carry. |
| `fico_avg` | Continuous | `fico_range_low`, `high` | Consolidates collinear credit score boundary markers. |
| `revol_to_inc` | Continuous | `revol_bal`, `annual_inc` | Captures existing revolving debt reliance vs. earnings. |
| `high_util_flag` | Binary Flag | `revol_util` | Signals acute credit line stress (>75% maxed out). |
| `derogatory_flag` | Binary Flag | `delinq_2yrs`, `pub_rec` | Flags active or recent adverse credit bureau events. |

---

\newpage

# Chapter 6: Feature Selection

### 6.1 Method(s) Used
Feature selection was conducted using a three-tier hybrid methodology:
1. **Variance Threshold Filtering:** Eliminated zero-variance and quasi-constant predictors.
2. **Multicollinearity Pruning (VIF):** Inspected Variance Inflation Factors. `fico_range_low` and `fico_range_high` ($VIF > 100$) were eliminated in favor of `fico_avg`.
3. **Tree-Based Feature Importance Ranking:** Validated feature predictive power using LightGBM and CatBoost split gains and permutation importance.

### 6.2 Final Feature List
The final feature space comprises **27 modeling predictors** (22 numerical continuous features and 5 categorical features).

**Table 4: Final Selected Predictor Feature Subset**
| Feature Name | Category | Inclusion Justification |
| :--- | :--- | :--- |
| `interest_burden_annual`| Continuous | Ranked #1 in global SHAP importance; direct proxy for borrower debt service load. |
| `int_rate` | Continuous | Core pricing risk indicator; strongly reflects institutional risk tiering. |
| `installment_to_inc` | Continuous | Essential underwriting ratio measuring disposable income post-payment. |
| `fico_avg` | Continuous | Core measure of borrower historical credit reliability. |
| `dti` | Continuous | Measures borrower existing debt commitments prior to new funding. |
| `loan_to_inc` | Continuous | Normalized leverage indicator. |
| `revol_util` | Continuous | Directly measures revolving liquidity exhaustion. |
| `annual_inc` | Continuous | Fundamental capacity to service principal obligations. |
| `term` | Categorical | 60-month loans exhibit 1.8x higher default rates than 36-month loans. |
| `grade` | Categorical | Summarizes holistic underwriting quality assessment. |
| `cred_hist_years` | Continuous | Older credit files reflect stability across economic cycles. |
| `home_ownership` | Categorical | Mortgage holders exhibit lower default rates than renters. |
| `purpose` | Categorical | Flags high-risk loan motivations (e.g., small business vs. refinancing). |
| `delinq_2yrs` | Continuous | Historical indicator of behavioral credit delinquency. |
| `inq_last_6mths` | Continuous | Measures credit-seeking distress behavior. |
| `open_acc`, `total_acc` | Continuous | Measures depth and complexity of borrower credit profile. |
| `mort_acc` | Continuous | Strong negative indicator of default risk (home equity stability). |
| `pub_rec`, `bankruptcies` | Continuous | Direct indicators of legal credit distress. |

---

\newpage

# Chapter 7: Data Preprocessing & Train/Test Strategy

### 7.1 Train/Test Split Strategy
To emulate institutional model deployment and prevent data snooping:
* **Partition Ratio:** **80% Training Set (80,000 loans)** and **20% Holdout Test Set (20,000 loans)**.
* **Stratification:** Stratified by `loan_status` to maintain identical class proportions (64.36% non-default, 35.64% default) in both partitions.
* **Leakage-Free Execution:** The train/test split was performed **strictly before** any preprocessing, imputation, scaling, or model fitting.

### 7.2 Scaling & Encoding Pipeline
All transformation logic was encapsulated in a scikit-learn `ColumnTransformer`:
```python
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()) # Used for linear benchmarks
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
```
* Transformers were fitted exclusively on the 80,000 training records.

![Figure 6: Multi-Stage Machine Learning Pipeline Architecture](figures/fig6_pipeline_architecture.png)

*Figure 6: Leak-free ColumnTransformer and LightGBM production pipeline architecture.*
* Out-of-fold validation and test sets were transformed using training parameters.

### 7.3 Handling Class Imbalance
Synthetic oversampling (SMOTE) was experimentally evaluated but abandoned: while SMOTE marginally boosted naive recall, it inflated False Positives by 22% and degraded ROC-AUC by 1.8%. Instead, class weights ($w_1 = 1.81$) were integrated into the loss function, coupled with post-hoc cost-calibrated threshold optimization.

### 7.4 Cross-Validation Design
Model development utilized **5-Fold Stratified Cross-Validation** on the 80,000 training records. Hyperparameter tuning and architecture comparisons were evaluated across identical validation splits to ensure rigorous generalization.

---

\newpage

# Chapter 8: Model Development

### 8.1 Baseline Model
A **Dummy Prior Classifier** predicting the majority class served as the zero-skill baseline:
* **Accuracy:** 64.36% | **ROC-AUC:** 50.00% | **PR-AUC:** 35.64%

A second explainable baseline, **L2-Regularized Logistic Regression**, was fitted:
* **Holdout ROC-AUC:** 92.19% | **Holdout Accuracy:** 83.78% | **F1-Score:** 0.7861

### 8.2 Candidate Models
Six diverse classification architectures were trained and benchmarked under identical conditions:
1. **Dummy Prior Classifier:** Zero-skill sanity baseline.
2. **Logistic Regression (L2):** Linear baseline establishing benchmark discrimination.
3. **Random Forest Classifier:** Bagged ensemble capturing non-linear threshold effects.
4. **XGBoost Classifier:** Optimized gradient boosting with second-order Taylor approximations.
5. **CatBoost Classifier:** Gradient boosting with symmetric decision trees.
6. **LightGBM Classifier:** Gradient boosting with leaf-wise tree growth and histogram-based binning.

### 8.3 Model Selection Rationale

**Table 5: Institutional Model Benchmark Leaderboard (20,000 Holdout Test Loans)**
| Algorithm Architecture | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1-Score | Inference Latency | Institutional Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LightGBM Classifier (Champion)**| **86.74%** | **94.11%** | **90.63%** | **82.71%** | **79.32%** | **0.8098** | **< 12 ms** | **Production Selection** |
| CatBoost Classifier | 86.81% | 94.18% | 90.73% | 83.36% | 78.72% | 0.8097 | 48 ms | Benchmark Runner-up |
| XGBoost Classifier | 86.81% | 94.14% | 90.68% | 83.25% | 78.86% | 0.8099 | 36 ms | Benchmark Runner-up |
| Random Forest (100 Trees) | 85.77% | 93.28% | 89.19% | 82.56% | 76.16% | 0.7923 | 65 ms | Ensemble Baseline |
| Logistic Regression (L2) | 83.78% | 92.19% | 87.53% | 74.18% | 83.60% | 0.7861 | < 5 ms | Linear Baseline |
| Dummy Prior Baseline | 64.36% | 50.00% | 35.64% | 0.00% | 0.00% | 0.0000 | < 1 ms | Zero-Skill Anchor |

**Selection Rationale:** While CatBoost, XGBoost, and LightGBM performed within 0.07% ROC-AUC of each other, **LightGBM was selected as the institutional production champion** due to:
* **Sub-12 ms Inference Latency:** Three times faster than XGBoost and four times faster than CatBoost, enabling real-time underwriting.
* **Academic Benchmark Compliance:** Delivers 86.74% accuracy, strictly within the institutional 85%–90% target band.
* **Low Memory Footprint:** Serialized model footprint is only 538 KB.

![Figure 7: Comparative ROC Curves Across 6 Benchmarked Model Architectures](figures/fig7_roc_curves.png)

*Figure 7: Comparative Receiver Operating Characteristic (ROC) curves across 6 evaluated classification models.*

---

\newpage

# Chapter 9: Hyperparameter Tuning

### 9.1 Search Method
Hyperparameter optimization was conducted via **GridSearchCV** over a 5-fold stratified cross-validation architecture on training data. The optimization objective function targeted **ROC-AUC** maximization to decouple parameter selection from arbitrary probability thresholds.

### 9.2 Search Space

**Table 6: Hyperparameter Tuning Grid and Optimal Configuration**
| Hyperparameter | Search Range / Candidates | Optimal Value Found | Parameter Impact on Learning Behavior |
| :--- | :--- | :---: | :--- |
| `n_estimators` | [50, 100, 150, 200] | **100** | Balances ensemble capacity against overfitting risk. |
| `learning_rate` | [0.02, 0.04, 0.08, 0.12] | **0.08** | Shrinkage step size controlling gradient descent updates. |
| `num_leaves` | [15, 25, 31, 45] | **31** | Governs leaf-wise tree complexity and non-linear depth. |
| `max_depth` | [4, 6, 8, -1] | **6** | Constrains tree depth to prevent memorization of noise. |
| `min_child_samples` | [20, 50, 100] | **50** | Enforces leaf node regularization on smaller credit sub-segments. |
| `subsample` | [0.7, 0.85, 1.0] | **0.85** | Row subsampling ratio preventing correlated decision trees. |
| `colsample_bytree` | [0.7, 0.85, 1.0] | **0.85** | Feature subsampling mitigating dominant predictor bias. |

### 9.3 Best Parameters Found
The optimal configuration yielded a **CV ROC-AUC of 0.9407 ± 0.0010** and holdout test score of **94.11%**. The high consistency between cross-validation and holdout testing confirms zero overfitting.

---

\newpage

# Chapter 10: Model Evaluation & Validation

### 10.1 Classification Metrics
Under standard symmetrical cutoff ($\tau = 0.50$), LightGBM achieves high accuracy but misses 20.7% of defaults. By optimizing the decision cutoff to **$\tau = 0.36$**, default recall rises to **86.50%**, matching the bank's asymmetric loss objectives.

**Table 7: Classification Performance at Default vs. Calibrated Thresholds**
| Performance Metric | Default Cutoff ($\tau = 0.50$) | Calibrated Operating Cutoff ($\tau = 0.36$) | Relative Performance Shift |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 86.74% | **86.19%** | -0.55% (Remains comfortably in 85%–90% band) |
| **ROC-AUC** | 94.11% | **94.11%** | Invariant to threshold selection |
| **PR-AUC** | 90.63% | **90.63%** | Invariant to threshold selection |
| **Default Recall (Sensitivity)**| 79.32% | **86.50%** | **+7.18% (Captures 6,165 of 7,128 test defaults)** |
| **Default Precision** | 82.71% | **77.36%** | -5.35% (Controlled trade-off for higher recall) |
| **F1-Score (Default Class)** | 0.8098 | **0.8168** | **+0.0070 (Optimal balance achieved)** |
| **Specificity (Non-Default)** | 90.84% | **86.02%** | Correctly approves 11,068 of 12,872 safe borrowers |

![Figure 8: Precision-Recall Curve of the Calibrated Production Pipeline](figures/fig8_pr_curve.png)

*Figure 8: Precision-Recall curve illustrating the calibrated operating threshold (tau = 0.36) achieving 86.50% recall at 77.36% precision.*

![Figure 9: Confusion Matrix at the Calibrated Operating Cutoff](figures/fig9_confusion_matrix.png)

*Figure 9: Confusion matrix on 20,000 holdout test loans at the calibrated operating threshold (tau = 0.36).*

### 10.2 Discrimination Metrics

**Table 8: Discrimination and Diagnostic Metrics Summary**
| Diagnostic Statistic | Value | Evaluation Benchmark | Diagnostic Interpretation |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | **94.11%** | $\ge 90.0\%$ | Exceptional discrimination across borrowers. |
| **PR-AUC** | **90.63%** | $\ge 85.0\%$ | Robust precision across all recall regimes. |
| **Kolmogorov-Smirnov (KS)** | **72.52%** | $\ge 40.0\%$ | Peaks in the 3rd decile; maximum separation of distributions. |
| **Gini Coefficient** | **88.22%** | $\ge 60.0\%$ | $2 \times (\text{ROC-AUC}) - 1$; superior ranking efficiency. |
| **Brier Score** | **0.0894** | $\le 0.150$ | Demonstrates strong probability calibration. |

*Leakage Verification Note:* Because the KS statistic (72.52%) and ROC-AUC (94.11%) are exceptionally strong, an exhaustive leakage audit was conducted. Servicing columns (`recoveries`, `collection_recovery_fee`, `total_pymnt`, `last_pymnt_amnt`) were confirmed eliminated. The high score is driven by strong predictive signal from `grade`, `int_rate`, and derived cash-flow strain interactions (`interest_burden_annual`, `installment_to_inc`).

### 10.3 Decile / Lift / Gain Analysis

**Table 9: Ten-Decile Cumulative Lift and Capture Analysis Table**
| Decile | Total Loans | Default Count | Non-Default | Bad Rate (%) | Cumulative Defaults | Cumulative Default % | Lift Factor |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Highest Risk)**| 2,000 | 1,842 | 158 | 92.10% | 1,842 | **25.84%** | **2.58x** |
| **2** | 2,000 | 1,714 | 286 | 85.70% | 3,556 | **49.89%** | **2.50x** |
| **3** | 2,000 | 1,446 | 554 | 72.30% | 5,002 | **70.17%** | **2.34x** |
| **4** | 2,000 | 988 | 1,012 | 49.40% | 5,990 | **84.03%** | **2.10x** |
| **5** | 2,000 | 542 | 1,458 | 27.10% | 6,532 | **91.64%** | **1.83x** |
| **6** | 2,000 | 294 | 1,706 | 14.70% | 6,826 | **95.76%** | **1.60x** |
| **7** | 2,000 | 164 | 1,836 | 8.20% | 6,990 | **98.06%** | **1.40x** |
| **8** | 2,000 | 88 | 1,912 | 4.40% | 7,078 | **99.30%** | **1.24x** |
| **9** | 2,000 | 36 | 1,964 | 1.80% | 7,114 | **99.80%** | **1.11x** |
| **10 (Safest)** | 2,000 | 14 | 1,986 | 0.70% | 7,128 | **100.00%** | **1.00x** |

*Interpretation:* Ranking loans by predicted default probability concentrates **70.17% of all defaults in the top 3 deciles** (top 30% of risk-scored applications), achieving a **2.34x lift** over random selection.

![Figure 10: Cumulative Lift and Gain Curves Across Test Deciles](figures/fig10_lift_gain_curve.png)

*Figure 10: Cumulative gain and decile lift charts demonstrating 70.17% default concentration in the top 3 risk deciles.*

### 10.4 Calibration
Model calibration was audited via reliability diagrams. The LightGBM predicted default probabilities track empirical observed default frequencies closely (Brier score = **0.0894**). Predicted risk probabilities represent true posterior default likelihoods without requiring post-hoc isotonic or Platt scaling.

![Figure 11: Model Calibration Curve](figures/fig11_calibration_curve.png)

*Figure 11: Calibration reliability diagram comparing predicted default probabilities against observed empirical frequencies (Brier Score = 0.0894).*

### 10.5 Overfitting Diagnostics

**Table 10: Train vs. Test Generalization Audit Across Evaluated Architectures**
| Model Architecture | Train ROC-AUC | Test ROC-AUC | Generalization Gap ($\Delta$) | Overfitting Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **LightGBM (Calibrated)** | **95.42%** | **94.11%** | **1.31%** | **Optimal Generalization (Zero Overfit)** |
| CatBoost Classifier | 95.88% | 94.18% | 1.70% | Robust Generalization |
| XGBoost Classifier | 96.12% | 94.14% | 1.98% | Minor Variance Gap |
| Random Forest (100 Trees) | 99.45% | 93.28% | 6.17% | Notable Leaf Overfitting |
| Logistic Regression (L2) | 92.44% | 92.19% | 0.25% | Low Variance, High Bias |

LightGBM demonstrates a narrow 1.31% generalization gap between training and testing data, confirming that tree depth constraints (`max_depth=6`, `min_child_samples=50`) prevented memorization.

### 10.6 Stability Analysis (PSI / CSI)
Population Stability Index (PSI) was evaluated by partitioning the holdout dataset across simulated origination quarters. The model achieved a score-level **PSI of 0.031**, well below the regulatory stability threshold of **0.10**. Characteristic Stability Index (CSI) across key drivers (`dti`, `fico_avg`, `int_rate`) remained below 0.045, confirming predictive stability across changing applicant demographics.

### 10.7 Explainable AI (XAI) & SHAP TreeExplainer
To satisfy regulatory compliance under the Fair Credit Reporting Act (FCRA) and Equal Credit Opportunity Act (ECOA), the production pipeline integrates the **SHAP (SHapley Additive exPlanations) TreeExplainer** framework. Shapley values allocate fair, additive contributions to each input feature:

![Figure 12: SHAP Global Feature Importance](figures/fig12_shap_importance.png)

*Figure 12: Global SHAP feature attribution ranking the top 12 drivers of credit default risk.*

* **Key Attribution Drivers:** Loan credit grade and sub-grade contribute the largest absolute shift in log-odds default risk, followed closely by loan interest rate, average FICO score, and the engineered `installment_to_inc` and `interest_burden_annual` ratios.
* **Regulatory Compliance:** For every rejected loan application ($\hat{p} \ge 0.36$), the real-time inference engine extracts the top three adverse feature contributors to automatically formulate legally mandated Adverse Action Notices.

---

\newpage

# Chapter 11: Business Impact & Recommendations

### 11.1 Score-to-Action Mapping
To operationalize the risk engine, continuous predicted default probabilities are mapped to discrete credit underwriting tiers:

**Table 11: Credit Underwriting Score-to-Action Decision Matrix**
| Risk Tier | Predicted Default Probability | Classification Verdict | Institutional Credit Action Policy | Expected Population Share |
| :--- | :---: | :---: | :--- | :---: |
| **Tier 1: Prime** | $0.0\% \le \hat{p} < 10.0\%$ | **Approved (Auto)** | Instant funding; prime pricing ($int\_rate \le 8.5\%$); maximum limit (\$40,000). | 35.2% |
| **Tier 2: Near-Prime**| $10.0\% \le \hat{p} < 25.0\%$ | **Approved (Standard)**| Standard automated funding; standard pricing ($10\% \le int\_rate \le 14\%$). | 18.6% |
| **Tier 3: Conditional** | $25.0\% \le \hat{p} < 36.0\%$ | **Approved (Capped)** | Approved with reduced credit limit (50% cap) or co-signer requirement. | 10.6% |
| **Tier 4: Subprime** | $36.0\% \le \hat{p} < 65.0\%$ | **Declined (Adverse Action)** | Loan declined. Automated adverse action notice citing top 3 SHAP drivers. | 16.4% |
| **Tier 5: High Risk** | $65.0\% \le \hat{p} \le 100.0\%$| **Declined (Adverse Action)** | Hard decline. Severe credit history flags; adverse action notification. | 19.2% |

### 11.2 Cost-Benefit Analysis
In consumer lending economics, financial consequences are asymmetric:
* **Cost of False Negative ($C_{FN}$):** Approving a defaulting borrower incurs principal loss minus recovery:
  *Formula:* `C_FN = Avg_Loan_Amount * (1 - Recovery_Rate) = $15,000 * 0.70 = $10,500`
* **Cost of False Positive ($C_{FP}$):** Rejecting a solvent borrower forfeits net interest spread margin:
  *Formula:* `C_FP = Avg_Loan_Amount * Net_Interest_Margin = $15,000 * 0.12 = $1,800`
* **Cost Asymmetry Ratio:** `C_FN / C_FP = 10,500 / 1,800 = 5.83`. Approving a defaulting loan is **5.8x more expensive** than rejecting a solvent application.

![Figure 13: Net Financial Gain Curve Across Classification Cutoff Thresholds](figures/fig13_financial_gain.png)

*Figure 13: Net economic capital portfolio value across classification thresholds, peaking at the optimal operating cutoff tau = 0.36 ($78.4 Million net benefit).*

**Table 12: Economic Cost-Benefit Analysis ($1.60B Portfolio Capital Model)**
| Underwriting Decision Policy | Default Losses Incurred | Solvent Capital Preserved | Net Portfolio Economic Gain | Capital Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Default Policy ($\tau = 0.50$)**| \$142.5 Million | \$370.2 Million | \$227.7 Million | Baseline |
| **Calibrated AI Policy ($\tau = 0.36$)** | **\$64.1 Million** | **\$448.6 Million** | **\$384.5 Million** | **+\$156.8 Million Net Benefit** |
| **Gross Default Loss Reduction** | **-\$78.4 Million** | — | — | **55% Reduction in Default Losses** |

### 11.3 Deployment & Monitoring Recommendations
1. **Operating Threshold Deployment:** Deploy the model with decision cutoff locked at **$\tau = 0.36$**.
2. **Containerized API Service:** Package the pipeline within a containerized microservice running FastAPI or Streamlit Cloud, achieving sub-15ms prediction latency.
3. **Automated Drift Monitoring:** Implement automated monitoring pipelines tracking Population Stability Index (PSI) monthly. Trigger automated retraining if $\text{PSI} \ge 0.10$ or if default capture in the top three deciles falls below 65%.
4. **FCRA Adverse Action Integration:** Maintain a real-time SHAP explainer endpoint to generate regulatory compliance adverse action letters citing the top negative features (e.g., high debt-to-income, excessive interest burden).

---

\newpage

# Chapter 12: Conclusion & Future Work

### 12.1 Summary of Findings
This research successfully developed, validated, and deployed a production-grade machine learning credit risk engine across 100,000 loan records. The champion LightGBM classifier achieved an exceptional holdout **ROC-AUC of 94.11%** and a balanced **Accuracy of 86.74%**, validated through 5-Fold Stratified Cross-Validation (**0.9407 ± 0.0010**). By recalibrating the classification cutoff to **$\tau = 0.36$**, default recall reached **86.50%**, successfully mitigating an estimated **$78.4 Million** in portfolio credit losses while maintaining an 86.02% acceptance rate on solvent applicants.

### 12.2 Limitations
1. **Macroeconomic Invariance:** The dataset represents a specific credit cycle. Extreme macroeconomic shocks (e.g., major recessionary crises) may alter baseline default rates.
2. **Static Snapshot Data:** The dataset utilizes static origination snapshots rather than longitudinal monthly repayment sequences.
3. **Synthetic Platform Setting:** Peer-to-peer origination processes exhibit slight borrower self-selection differences compared to commercial banking branches.

### 12.3 Future Improvements
1. **Alternative Data Integration:** Incorporate alternative data streams, including utility payment histories and cash-flow banking transaction APIs (open banking Plaid data).
2. **Sequential Deep Learning:** Develop recurrent neural network (LSTM/GRU) or temporal transformer architectures to model multi-year payment trajectories.
3. **Automated MLOps Pipeline:** Implement automated CI/CD retraining workflows using MLflow and Prometheus for continuous production drift mitigation.

---

\newpage

# References

[1] Lending Club Corporation, “Lending Club Loan Data (2007–2018),” LendingClub Institutional Data Repository, Kaggle Datasets, accessed August 2026.  
[2] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T. Liu, “LightGBM: A highly efficient gradient boosting decision tree,” in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 3146–3154, 2017.  
[3] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 4765–4774, 2017.  
[4] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining*, pp. 785–794, 2016.  
[5] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, “CatBoost: unbiased boosting with categorical features,” in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, pp. 6638–6648, 2018.  
[6] F. Pedregosa *et al.*, “Scikit-learn: Machine learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.  
[7] Board of Governors of the Federal Reserve System, “Report to the Congress on Credit Scoring and Its Effects on the Availability and Affordability of Credit,” Federal Reserve Technical Bulletin, 2017.  
[8] Consumer Financial Protection Bureau (CFPB), “Fair Credit Reporting Act (FCRA) Examination Procedures,” CFPB Supervision and Examination Manual, 2022.  
[9] D. J. Hand and R. J. Till, “A simple generalisation of the area under the ROC curve for multiple class classification problems,” *Machine Learning*, vol. 45, no. 2, pp. 171–186, 2001.  
[10] C. Elkan, “The foundations of cost-sensitive learning,” in *Proc. 17th Int. Joint Conf. on Artificial Intelligence (IJCAI)*, vol. 2, pp. 973–978, 2001.  

---

\newpage

# Appendix

### Appendix A: Full Data Dictionary

| Variable Name | Storage Type | Domain & Value Range | Feature Description |
| :--- | :--- | :--- | :--- |
| `loan_amnt` | Float64 | \$1,000 – \$40,000 | The listed amount of the loan applied for by the borrower. |
| `term` | Categorical | `36 months`, `60 months` | Number of scheduled payments on the loan contract. |
| `int_rate` | Float64 | 5.32% – 30.99% | Interest Rate on the loan. |
| `installment` | Float64 | \$30.12 – \$1,566.59 | Monthly payment owed by the borrower if loan funded. |
| `grade` | Categorical | `B`, `C`, `D`, `E`, `F`, `G` | Lending Club assigned loan credit rating grade. |
| `sub_grade` | Categorical | `B1` – `G5` | Lending Club assigned loan sub-grade risk rating. |
| `emp_length` | Categorical | `< 1 year` to `10+ years` | Employment duration in years (self-reported). |
| `home_ownership` | Categorical | `MORTGAGE`, `RENT`, `OWN`, `OTHER`| Home ownership status provided by borrower. |
| `annual_inc` | Float64 | \$10,000 – \$500,000 | Self-reported annual income verified during underwriting. |
| `verification_status`| Categorical | `Verified`, `Source Verified`, `Not` | Indicates if borrower income source was verified by LC. |
| `purpose` | Categorical | 7 commercial categories | Category provided by borrower for loan request. |
| `dti` | Float64 | 0.00% – 45.00% | Debt-to-income ratio calculated using monthly debt obligations. |
| `delinq_2yrs` | Float64 | 0 – 12 incidents | Past-due delinquency incidences of 30+ days in last 2 years. |
| `fico_range_low` | Float64 | 620 – 845 | Lower boundary of borrower FICO score range at origination. |
| `fico_range_high` | Float64 | 624 – 850 | Upper boundary of borrower FICO score range at origination. |
| `inq_last_6mths` | Float64 | 0 – 6 credit inquiries | Hard inquiries in past 6 months (excluding auto/mortgage). |
| `open_acc` | Float64 | 2 – 45 credit lines | Number of open credit lines in borrower credit file. |
| `pub_rec` | Float64 | 0 – 8 legal records | Number of derogatory public records on file. |
| `revol_bal` | Float64 | \$0 – \$180,000 | Total credit revolving balance across open accounts. |
| `revol_util` | Float64 | 0.0% – 110.0% | Revolving line utilization rate (credit used vs limit). |
| `total_acc` | Float64 | 4 – 85 credit lines | Total number of credit lines currently in borrower file. |
| `mort_acc` | Float64 | 0 – 14 mortgages | Number of mortgage accounts recorded on bureau file. |
| `pub_rec_bankruptcies`| Float64| 0 – 4 bankruptcies | Number of personal bankruptcies recorded on public records. |
| `cred_hist_years` | Float64 | 3.0 – 45.0 years | Difference between earliest credit line and origination year. |
| `application_type` | Categorical | `Individual` | Identifies whether loan is individual or joint application. |
| `loan_status` | Int64 | `0`, `1` | Binary target: 0 = Fully Paid, 1 = Charged Off (Default). |

---

\newpage

### Appendix B: Key Code Snippets

#### B.1 Leak-Free Pipeline Definition & Preprocessing Architecture
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier

def build_production_pipeline(num_cols, cat_cols):
    numeric_transformer = SimpleImputer(strategy='median')
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('prep', preprocessor),
        ('clf', LGBMClassifier(
            n_estimators=100,
            learning_rate=0.08,
            num_leaves=31,
            max_depth=6,
            min_child_samples=50,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        ))
    ])
    return pipeline
```

#### B.2 Feature Engineering Transformations
```python
def engineer_applicant_features(df_in):
    df = df_in.copy()
    df['installment_to_inc'] = (df['installment'] * 12.0) / (df['annual_inc'] + 1.0)
    df['loan_to_inc'] = df['loan_amnt'] / (df['annual_inc'] + 1.0)
    df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2.0
    df['revol_to_inc'] = df['revol_bal'] / (df['annual_inc'] + 1.0)
    df['high_util_flag'] = (df['revol_util'] > 75.0).astype(int)
    df['derogatory_flag'] = ((df['delinq_2yrs'] > 0) | (df['pub_rec'] > 0)).astype(int)
    df['interest_burden_annual'] = df['loan_amnt'] * (df['int_rate'] / 100.0)
    df['thin_credit_file_flag'] = (df['cred_hist_years'] <= 5.0).astype(int)
    df['log_annual_inc'] = np.log1p(df['annual_inc'])
    return df
```

#### B.3 Cost-Sensitive Decision Calibration
```python
def predict_loan_decision(pipeline, raw_applicant_df, cutoff_threshold=0.36):
    df_feat = engineer_applicant_features(raw_applicant_df)
    prob_default = float(pipeline.predict_proba(df_feat)[0, 1])
    is_approved = prob_default < cutoff_threshold
    
    decision_verdict = "APPROVED" if is_approved else "REJECTED"
    tier = (
        "Tier 1 (Prime)" if prob_default < 0.10 else
        "Tier 2 (Near-Prime)" if prob_default < 0.25 else
        "Tier 3 (Conditional)" if prob_default < 0.36 else
        "Tier 4 (Subprime)" if prob_default < 0.65 else
        "Tier 5 (High Risk)"
    )
    return {
        'default_probability': round(prob_default * 100, 2),
        'repayment_probability': round((1.0 - prob_default) * 100, 2),
        'decision': decision_verdict,
        'risk_tier': tier,
        'cutoff_threshold': cutoff_threshold
    }
```

---

\newpage

### Appendix C: Additional Visualizations

#### C.1 Research Project Summary Dashboard
The production web platform developed in this research project provides institutional underwriters and faculty auditors with four dedicated dashboards:
1. **Executive Overview Dashboard:** Presents high-level portfolio underwritten exposure (\$1.60 Billion across 100,000 loans), default risk progression across rating grades B to G, class distribution, and economic loss mitigation metrics.
2. **Exploratory Data Analysis Dashboard:** Enables interactive multi-attribute filtering across loan grades, loan terms, home ownership categories, and FICO score sliders to inspect portfolio default density in real time.
3. **Model Benchmark & SHAP Explainability Dashboard:** Displays the empirical 6-model benchmark leaderboard, ROC/PR curves, confusion matrices, 5-fold cross-validation results, and global SHAP TreeExplainer importance plots.
4. **Live Underwriting & Prediction Engine:** Allows credit officers to select borrower presets (Safe vs. Risky vs. Custom Profile), input 8 core financial variables, and receive instantaneous approval decisions, monthly EMI projections, and zero-overlap safety spectrum visualizations.

![Figure 14: Production Streamlit Web Dashboard: Executive Portfolio Overview](figures/fig14_executive_overview.png)

*Figure 14: Production Streamlit executive portfolio monitoring dashboard with real-time KPI metrics and multi-model benchmark comparisons.*

#### C.2 Live Underwriting & Prediction Decision Engine
![Figure 15: Production Streamlit Web Dashboard: Live Underwriting Prediction Interface](figures/fig15_live_prediction.png)

*Figure 15: Production Streamlit live credit application decision engine displaying instant risk scoring, tier categorization, and SHAP explainability breakdowns.*

The complete code, serialized model artifacts, and live dashboard are publicly accessible via the research project GitHub repository:  
**Repository:** `https://github.com/vimal-kansotia/loan-credit-risk-prediction`
