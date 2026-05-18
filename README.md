# House Price Prediction (Advanced)

**House-Price-Prediction-Advanced** is a machine learning web application that predicts house prices based on key features using a Random Forest model. This project includes thorough data analysis, interactive visualizations, and a user-friendly Flask web interface.

##  Project Overview

Predicting house prices is a classic data science problem with real-world value for buyers, sellers, and developers. This project demonstrates the full cycle of a machine learning workflow:

- **Data Exploration**
- **EDA Visualizations**
- **Feature Engineering**
- **Model Training (Random Forest)**
- **Model Evaluation and Interpretation**
- **Web Deployment (Flask App)**
- **User-friendly Prediction Interface**

## Workflow Diagram

Here’s a high-level workflow for the project:

```
 +--------------+        +-------------------+       +----------------------+ 
 | house_data   | -----> |  train_model.py   | --->  | random_forest_model |
 +--------------+        +-------------------+       +----------------------+
        |                        |                               |
        |                        |          +--------------------+
        |                        +------->  |  images/           | (EDA/Model Plots)
        |                                   +--------------------+
        |
 +-----------------+       User Input        +-------------------+
 | Flask Web App   | <---------------------> | index.html        |
 | (app/app.py)    | --------------------->  |  style.css        |
 +-----------------+       Prediction        +-------------------+
```


##  Features

- **Clean EDA & Visualizations**: Automated EDA with plots:
  - ![Correlation Heatmap](images/correlation_heatmap.png)
  - ![Feature Importance](images/feature_importance.png)
  - ![Prediction Results](images/prediction_graph.png)
- **Model Training Script**: Ready-to-run `train_model.py` builds and saves the model.
- **Interactive Web App**: Enter features, get instant price predictions via Flask + HTML.
- **Reusable Notebook**: Jupyter notebook for in-depth experiments.
- **Modular Project Layout**: Easy to modify and extend.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/House-Price-Prediction-Advanced.git
cd House-Price-Prediction-Advanced
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run model training (generates images and model)

```bash
python train_model.py
```

### 4. Start the web app

```bash
python app/app.py
```

Visit: [https://localhost:5000](https://localhost:5000)

## Example Results (Visuals)

### Correlation Heatmap
![Correlation Heatmap](<img width="632" height="585" alt="58945" src="https://github.com/user-attachments/assets/0cdb9541-8acf-48f8-9192-af7ca80f832a" />
/correlation_heatmap.png)

### Feature Importance
![Feature Importance](<img width="647" height="470" alt="58946" src="https://github.com/user-attachments/assets/50ac0b1e-72fb-4000-9b56-1491a04bf971" />
/feature_importance.png)

### Actual vs. Predicted Prices
![Prediction Graph]
(<img width="644" height="470" alt="58947" src="https://github.com/user-attachments/assets/84538062-47b0-4cac-b7cb-aa02781bc06b" />
/prediction_graph.png)


## Sample User Interface

> You can enter features like Area, Bedrooms, Bathrooms, Stories, and Parking on the main page, and instantly see the predicted price.


##  Reference: Jupyter Notebook

Check `notebooks/house_price_prediction.ipynb` for:
- Data cleaning
- Exploratory Data Analysis
- In-depth model evaluation


## Tech Stack

- Python, Flask
- scikit-learn, pandas, numpy
- matplotlib, seaborn
- HTML/CSS


## License

MIT License. See [LICENSE](LICENSE) for details.


## Acknowledgments

- [Kaggle (House Price Data)](https://www.kaggle.com/)
- scikit-learn documentation
