# CS4824 Project: Heart Failure Mortality Prediction

## Project Overview

This project builds a machine learning model to predict the likelihood of a death event in heart failure patients using clinical input features such as age, ejection fraction, serum creatinine, serum sodium, blood pressure status, diabetes status, and smoking status.

The main model used in this project is a **logistic regression model implemented from scratch** with NumPy. The project manually implements:

- sigmoid activation
- gradient descent training
- probability prediction
- binary classification using a threshold
- evaluation metrics such as accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrix counts

The project also includes a simple Flask web application where a user can enter patient values and receive a predicted probability of a death event.

## Dataset

The project uses heart failure clinical records stored in:

- [data/heart_failure_clinical_records_dataset.csv](/Users/sanjays/Desktop/VT/ML/CS4824_Project/data/heart_failure_clinical_records_dataset.csv)
- [data/heart_failure_clinical_records_5000.csv](/Users/sanjays/Desktop/VT/ML/CS4824_Project/data/heart_failure_clinical_records_5000.csv)

The final notebook and web app are configured to use the larger `heart_failure_clinical_records_5000.csv` file.

Target variable:

- `DEATH_EVENT`
  - `0` = patient survived during follow-up
  - `1` = patient died during follow-up

## What The Code Does

The pipeline used in the notebook and app is:

1. Load the dataset
2. Remove duplicate rows
3. Check for missing values and drop them if necessary
4. Split the data into training and testing sets
5. Standardize the continuous features
6. Train logistic regression from scratch
7. Evaluate the model on the test set
8. Compare the scratch model against a scikit-learn logistic regression baseline
9. Predict the probability of death for a custom patient input

## Model Configuration

The tuned configuration used in the final version is:

- learning rate: `0.03`
- epochs: `5000`
- classification threshold: `0.50`

## Current Results

The tuned model currently reports the following evaluation results on the test split:

- Accuracy: `0.8712`
- Precision: `0.7846`
- Recall: `0.7183`
- F1 score: `0.7500`
- ROC-AUC: `0.8875`
- Confusion matrix: `TN=179, FP=14, FN=20, TP=51`

## Files

- [heart_failure_demo.ipynb](/Users/sanjays/Desktop/VT/ML/CS4824_Project/heart_failure_demo.ipynb): main notebook containing preprocessing, training, evaluation, and custom-patient prediction
- [app.py](/Users/sanjays/Desktop/VT/ML/CS4824_Project/app.py): Flask app for the frontend demo
- [templates/index.html](/Users/sanjays/Desktop/VT/ML/CS4824_Project/templates/index.html): HTML template for the web app
- [requirements.txt](/Users/sanjays/Desktop/VT/ML/CS4824_Project/requirements.txt): Python dependencies

## How To Run The Notebook

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Then open:

- [heart_failure_demo.ipynb](/Users/sanjays/Desktop/VT/ML/CS4824_Project/heart_failure_demo.ipynb)

Run the cells from top to bottom.  
The final cell contains a `demo_patient` dictionary that you can edit to generate a new prediction.

## How To Run The Web App

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the Flask app:

```bash
python app.py
```

Then open this URL in your browser:

```text
http://127.0.0.1:5000
```

In the app, enter patient values into the form and submit to get:

- predicted probability of death event
- predicted class output

## Notes

- The logistic regression model is implemented manually and does not use `scikit-learn`.
- The notebook contains the full evaluation code.
- This README was formatted by ChatGPT but the content was human created.
