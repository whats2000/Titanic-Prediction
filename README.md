# Titanic Prediction

A machine learning project for predicting Titanic passenger survival using the famous Titanic dataset. This project demonstrates data preprocessing, feature engineering, and model training with XGBoost.

## Overview

This project uses the Titanic dataset from Kaggle to build a predictive model that determines whether a passenger survived the Titanic disaster based on features like age, sex, class, and more. The model is trained using XGBoost and optimized with hyperparameter tuning.

## Dataset

The dataset is sourced from the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) competition on Kaggle. It includes:

- `train.csv`: Training data with survival labels
- `test.csv`: Test data without survival labels
- `gender_submission.csv`: Sample submission file

Please download the dataset from Kaggle and place the CSV files in the project `data/` directory.

## Features

- Data loading and exploration
- Handling missing values and categorical encoding
- Feature selection and preprocessing
- XGBoost model training
- Hyperparameter tuning with RandomizedSearchCV
- Model evaluation and prediction
- Submission file generation

## Installation

### Prerequisites

- Python 3.10 or higher
- uv (for dependency management)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/whats2000/Titanic-Prediction.git
   cd Titanic-Prediction
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   uv run python --version
   ```

## Usage

### Running the Notebook

1. Launch Jupyter Notebook:
   ```bash
   uv run jupyter notebook
   ```

2. Open `titanic-prediction.ipynb` and run the cells sequentially.

### Key Steps in the Notebook

1. **Load Data**: Import and explore the Titanic dataset
2. **Preprocess Data**: Handle missing values, encode categorical variables
3. **Train Model**: Train an XGBoost classifier
4. **Hyperparameter Tuning**: Optimize model parameters using RandomizedSearchCV
5. **Evaluate Model**: Assess performance on validation data
6. **Make Predictions**: Generate predictions for the test set
7. **Create Submission**: Save predictions to `submission.csv`

### Direct Execution

You can also run the notebook cells directly using nbconvert:

```bash
uv run jupyter nbconvert --to notebook --execute titanic-prediction.ipynb
```

## Dependencies

- `ipykernel`: Jupyter kernel for Python
- `notebook`: Jupyter Notebook
- `numpy`: Numerical computing
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning library
- `tqdm`: Progress bars
- `xgboost`: Gradient boosting framework

## Model Performance

The model achieves an AUC score on the validation set. Hyperparameter tuning improves performance through RandomizedSearchCV with 10-fold cross-validation.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kaggle for providing the Titanic dataset
- The open-source community for the libraries used

## Kaggle Kernel

For direct use on Kaggle, you can copy and edit the code from this Kaggle notebook: [XGBoost Hands-on Practice with XGBoost](https://www.kaggle.com/code/whats2000/xgboost-hand-on-practice-with-xgboost#Load-the-data-and-display-head)
