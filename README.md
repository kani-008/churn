"""
# Customer Churn Prediction

This project uses an XGBoost classifier to predict customer churn with high accuracy. It includes data preprocessing, model training with GridSearchCV, and a Streamlit web app for real-time predictions.

## How to Run
1. Place your dataset in the `data/` folder (e.g., `telco_churn.csv`)
2. Install dependencies:
    pip install -r requirements.txt
3. Preprocess the data and train the model:
    Run `src/model_training.py`
4. Launch the app:
    streamlit run app/app.py

## Project Structure
- `data/`: Raw dataset
- `src/`: Scripts for preprocessing and training
- `app/`: Streamlit interface and saved model
- `assets/`: Visualizations and flowcharts

"""