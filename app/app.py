from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Adjust path for model loading
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest_model.pkl')
model = joblib.load(model_path)

# Feature columns based on data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'house_data.csv')
columns = pd.read_csv(data_path, nrows=1).drop('Price', axis=1).columns.tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            input_data = [float(request.form.get(col, 0)) for col in columns]
            df = pd.DataFrame([input_data], columns=columns)
            prediction = round(model.predict(df)[0], 2)
        except Exception as e:
            prediction = f"Invalid input: {e}"
    return render_template('index.html', columns=columns, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
