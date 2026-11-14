
# CloudPredict ☁️

## Intelligent Storage Forecasting System

CloudPredict is a machine-learning-based application that analyzes historical cloud storage usage and predicts future storage requirements.

## Project Objective

The main objective of CloudPredict is to help organizations estimate future storage consumption and plan their storage capacity in advance.

## Problem Statement

Organizations store increasing amounts of data in cloud storage. Without proper planning, they may face storage shortages. CloudPredict helps predict future storage requirements using historical data.

## Features

- Upload historical storage data using CSV.
- Analyze storage consumption.
- Display historical storage graphs.
- Predict future storage requirements.
- Compare three machine learning models.
- Evaluate models using MAE and RMSE.
- Generate storage capacity warnings.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

## Machine Learning Algorithms

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression

## Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

Models are evaluated using chronological training and testing data.

## Dataset Format

The CSV file must contain two columns:

- Date
- Storage_Used_GB

Example:

Date,Storage_Used_GB
2026-01-01,100
2026-01-02,105

The sample dataset contains simulated storage usage data.

## How to Run the Project

1. Install Python.
2. Clone or download this repository.
3. Install the required libraries:

   pip install -r requirements.txt

4. Run the application:

   python -m streamlit run app.py

5. Open the local URL displayed in the terminal.

## Future Scope

- Integration with AWS S3.
- Integration with Microsoft Azure.
- Integration with Google Cloud Storage.
- Real-time storage monitoring.
- Automated capacity recommendations.

## Project Information

Course: B.Tech CSE - Data Science

University: Shri Vaishnav Vidyapeeth Vishwavidyalaya

Academic Session: 2026-2027

Project Domain: Cloud Computing

## Disclaimer

This is an academic mini-project prototype. It currently uses CSV-based historical data and does not directly connect to cloud storage platforms.
