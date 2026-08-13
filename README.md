# Invoice Intelligence ML Project

This project provides machine learning models for invoice intelligence, including freight cost prediction and invoice flagging for potential risks.

## Project Structure

```
Invoice Intelligence ML project/
├── Data/
│   └── inventory.db                    # SQLite database with invoice data
├── Freight_cost_prediction/
│   ├── data_preprocessing.py          # Data loading and preprocessing for freight prediction
│   ├── model_evaluation.py            # Model training and evaluation functions
│   ├── train.py                       # Script to train freight prediction models
│   ├── predict.py                     # Script to make freight predictions
│   └── models/
│       └── predict_freight_model.pkl  # Trained freight prediction model
├── invoice_flagging/
│   ├── data_preprocessing.py          # Data loading and preprocessing for invoice flagging
│   ├── model_evaluation.py            # Model training and evaluation functions
│   ├── train.py                       # Script to train invoice flagging model
│   └── models/
│       ├── predict_flag_invoice.pkl   # Trained invoice flagging model
│       └── scaler.pkl                 # Feature scaler for invoice flagging
├── inference/
│   ├── predict_freight.py             # Inference script for freight prediction
│   └── predict_invoice_flag.py        # Inference script for invoice flagging
├── Notebooks/
│   └── Predicting Freight Cost.ipynb  # Jupyter notebook for data exploration
├── models/                            # Shared models directory
├── app.py                             # Streamlit web application
└── README.md                          # This file
```

## Features

- **Freight Cost Prediction**: Predicts freight costs based on invoice dollars and quantity using Linear Regression, Decision Tree, and Random Forest models.
- **Invoice Flagging**: Identifies potentially risky invoices based on discrepancies between invoice totals and item-level data, using a Random Forest classifier.
- **Web Interface**: Streamlit app for easy prediction without coding.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Required packages: pandas, scikit-learn, joblib, streamlit, sqlite3

### Installation

1. Clone or download the project repository.

2. Install dependencies:
   ```
   pip install pandas scikit-learn joblib streamlit
   ```

3. Ensure the database file `Data/inventory.db` is present.

## Usage

### Training Models

#### Freight Cost Prediction
```bash
cd Freight_cost_prediction
python train.py
```

#### Invoice Flagging
```bash
cd invoice_flagging
python train.py
```

### Making Predictions

#### Freight Cost
```bash
cd Freight_cost_prediction
python predict.py  # Uses example values
```

#### Invoice Flagging
```bash
cd inference
python predict_invoice_flag.py  # Uses example values
```

### Running the Web App

1. Ensure models are trained and saved.
2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
3. Open the provided URL in your browser (usually `http://localhost:8501`).

## Model Details

### Freight Prediction
- **Features**: Invoice dollars, quantity
- **Target**: Freight cost
- **Models**: Linear Regression (best performer), Decision Tree, Random Forest
- **Metrics**: MAE ~24.46, RMSE ~124.43, R² ~97.00%

### Invoice Flagging
- **Features**: Invoice quantity, dollars, freight, total item quantity, dollars
- **Target**: Flag (0 = normal, 1 = flagged)
- **Model**: Random Forest Classifier
- **Metrics**: Accuracy ~88%, Precision/Recall/F1 balanced

## Data

The project uses a SQLite database (`inventory.db`) containing tables:
- `vendor_invoice`: Invoice details
- `purchases`: Item-level purchase data
- Other related tables

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make changes and test.
4. Submit a pull request.

## License

This project is for educational purposes. Please check data licensing if using real datasets.

## Contact

For questions or issues, please open an issue in the repository.