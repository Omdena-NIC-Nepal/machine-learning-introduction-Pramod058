# Predicting House Prices Using Linear Regression

## Overview
This project demonstrates the application of Linear Regression for predicting house prices using the Boston Housing Dataset. It involves data exploration, preprocessing, model building, evaluation, and feature engineering to improve performance. The project is structured to guide students through the key aspects of supervised learning while ensuring proper collaboration and version control.

## Project Structure
```
├── data
│   └── boston_housing.csv
│   ├── only_x.csv
│   ├── only_ycsv
│   ├── X_test.csv
│   ├── X_train.csv
│   ├── y_pred_df.csv
│   ├── y_test.csv
│   ├── y_train.csv
│
├── notebooks
│   ├── t1_EDA.ipynb
│   ├── t2_Data_preprocessing.ipynb
│   ├── t3_Model_Training.ipynb
│   ├── t4_Model_Evaluation.ipynb
│   ├── t5_Feature_Engineering.ipynb
│
├── scripts
│   ├── data_preprocessing.py
│   ├── evaluate_model.py
│   ├── train_model.py
│
├── proj_Instruction.md
├── README.md
├── requirements.txt
├── .gitignore
```

## Steps to View and Run the Code

### 1. Clone the Repository
```
git clone https://github.com/Omdena-NIC-Nepal/machine-learning-introduction-Pramod058.git

```

### 2. Install Dependencies
Ensure you have Python installed, then run:
```
pip install -r requirements.txt
```

### 3. Explore the Data
Navigate to the `notebooks` directory and open `EDA.ipynb` to analyze dataset structure and insights.

### 4. Data Preprocessing
Run `Data_Preprocessing.ipynb` or execute the script:
```
python scripts/data_preprocessing.py
```

### 5. Model Training
Run `Model_Training.ipynb` to train the linear regression model, or use:
```
python scripts/train_model.py
```

### 6. Model Evaluation
Evaluate the trained model using `Model_Evaluation.ipynb` or execute:
```
python scripts/evaluate_model.py
```

### 7. Feature Engineering 
Improve the model by testing new features in `Feature_Engineering.ipynb`.




## Conclusion
This project provides a step-by-step implementation of Linear Regression for house price prediction. The insights and findings are included in the notebooks. Feel free to explore, modify, and improve the model!

## Author
- [Pramod Aryal](https://www.linkedin.com/in/pramod58/)

---
For any questions or improvements, please reach out or open an issue in the repository.