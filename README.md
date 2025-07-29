# Fraud Detection on PaySim Dataset

## Project Overview

This project focuses on detecting fraudulent transactions in a synthetic mobile money transfer dataset (PaySim). Leveraging machine learning with state-of-the-art tree-based models, the goal is to accurately identify rare fraud events, balancing the challenges of severe class imbalance and highly imbalanced transactional data.

---

## Features and Approach

- **Data Cleaning & Exploration:**  
  Comprehensive handling of missing values, duplicates, outliers and feature consistency checks to ensure data quality.

- **Feature Engineering:**  
  - One-hot encoding of transaction types  
  - Deriving numeric balance delta features (`deltaOrig`, `deltaDest`)  
  - Binary flags for zero balances, which are highly indicative of fraudulent behavior

- **Handling Data Imbalance:**  
  Using stratified sampling to downsize training data (100k stratified samples) combined with class weighting (`scale_pos_weight`) in XGBoost to address the rare-event problem efficiently and effectively.

- **Model Development:**  
  XGBoost classifier trained with carefully tuned hyperparameters, optimized for rare-event detection.

- **Evaluation & Explainability:**  
  - Metrics: ROC AUC and Precision-Recall AUC (AUPRC) for true rare-event performance   
  - Visualizations: Precision-Recall and ROC curves  
  - Feature Importance and SHAP values for interpretable fraud drivers  

- **Business Insights:**  
  Recommendations for real-time fraud prevention controls and monitoring strategy based on model findings and domain knowledge.

---

## Getting Started

### Prerequisites

- Python 3.9 or above  
- pip package manager

### Installation

1. Clone the repository:
git clone https://github.com/yourusername/fraud-detection-paysim.git
cd fraud-detection-paysim


2. (Recommended) Create and activate a virtual environment:
python -m venv fraud_env
source fraud_env/bin/activate # Linux/macOS
fraud_env\Scripts\activate # Windows



3. Install dependencies:
pip install -r requirements.txt



4. Download or place the full `Fraud.csv` in the `data/` directory, or use the provided `Fraud_sample.csv` for demonstration.

### Usage

- Open the Jupyter notebook `analysis.ipynb` in VS Code or Jupyter environment.
- Execute cells sequentially to reproduce analysis, model training, evaluation, and interpretation.
- Modify parameters or data paths as needed for your environment.

---

## Project Structure

fraud-detection-project/
├── analysis.ipynb # Main analysis and modeling notebook
├── Fraud_sample.csv # Small sample of dataset for demo/testing
├── requirements.txt # Python package dependencies
├── README.md # This file
├── data/ # (Optional) Directory for full dataset (not versioned)
└── .gitignore # Git ignore rules


---

## Results Snapshot

- **ROC AUC:** ~0.997 (reflecting highly accurate fraud detection)  
- **Precision-Recall AUC:** >0.80 (robust rare-event performance)  
- **Top Fraud Predictors:** Transaction types (TRANSFER, CASH_OUT), zero-balance flags, large balance deltas

Visualizations and detailed performance metrics are available within the notebook.

---

## Recommendations & Next Steps

- Productionize model with monitoring on model concept drift (Population Stability Index, SHAP drift).  
- Enforce business controls such as real-time transaction blocking on suspicious patterns identified by model.  
- Expand to hyperparameter tuning or ensemble approaches for potential further gains.  

---

## License

This project is for educational and internship use. Please do not redistribute proprietary data used herein.

---

## Contact

Developed by SAMARTH GUPTA | [sg9971973@gmail.com] | [https://github.com/PEKKAGAMING21]

---

Thank you for your interest in this fraud detection project! Please raise issues or pull requests for questions, suggestions, or improvements.