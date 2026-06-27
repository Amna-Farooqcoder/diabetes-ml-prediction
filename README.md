# diabetes-ml-prediction
Diabetes prediction ML model using Random Forest. Achieves 76.62% accuracy on the Pima Indians dataset. Built with Python, pandas, scikit-learn, and matplotlib.
# Diabetes Prediction ML Model

A machine learning model that predicts whether a patient has diabetes based on medical measurements.

## Results
- **Accuracy: 76.62%**
- Algorithm: Random Forest Classifier
- Dataset: Pima Indians Diabetes Dataset (768 patients)

## What I built
- Loaded and explored real medical data using pandas
- Cleaned missing values (fake zeros in Glucose, BMI, BloodPressure)
- Trained a Random Forest model using scikit-learn
- Visualized feature importance using matplotlib

## Key Finding
Glucose level is the strongest predictor of diabetes, followed by BMI and Age.

## Technologies
Python | pandas | scikit-learn | matplotlib

## How to run
```bash
pip install pandas scikit-learn matplotlib
python diabetes.py
```

## Author
Amna Farooq — BSc Mathematics with AI, UET Lahore
