import numpy as np
import pandas as pd
import datetime as dt
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')
fet = pd.read_csv('all_features.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]

    if features[3] == '0':
        features[3] = False
    else:
        features[3] = True

    df = fet[(fet['Store'] == int(features[0])) & (fet['IsHoliday'] == features[3]) & (fet['Date'] == features[2])]
    f_features = []
    d = dt.datetime.strptime(features[2], '%Y-%m-%d')
    c = 0

    if not df.empty:
        if df['Type'].iloc[0] == 'C':
            c = 1
        else:
            c = 0
    else:
        c = 0  # Handle the case when df is empty

    if df.shape[0] == 1:
        f_features.append(df['CPI'].iloc[0])
        f_features.append(d.date().day)
        f_features.append(int(features[1]))
        f_features.append(df['Fuel_Price'].iloc[0])
        f_features.append(int(features[3]))  # Use int instead of boolean
        f_features.append(d.date().month)
        f_features.append(df['Size'].iloc[0])
        f_features.append(int(features[0]))
        f_features.append(df['Temperature'].iloc[0])
        f_features.append(c)
        f_features.append(df['Unemployment'].iloc[0])
        f_features.append(d.date().year)

    final_features = np.array([f_features])  # Remove the extra list around f_features

    output = model.predict(final_features)[0]
    return render_template('index.html', output=output)

if __name__ == "__main__":
    app.run(debug=True)
