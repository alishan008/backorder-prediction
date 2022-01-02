from flask import Flask, render_template, request
from joblib import load
import pandas as pd
import numpy as np

app = Flask(__name__)

#scaler = load('scaler.joblib')
ml_model = load('model.joblib')

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    print("I was here 1")
    if request.method == 'POST':
        print(request.form.get('nationalinv'))
        try:
            national_inv = float(request.form['nationalinv'])
            lead_time = float(request.form['leadtime'])
            in_transit_qty = float(request.form['intraqty'])
            forecast_3_month = float(request.form['3mnthforecast'])
            sales_1_month = float(request.form['1mthsales'])
            pieces_past_due = float(request.form['pcspastdue'])
            perf_6_month_avg = float(request.form['6mthavg'])
            local_bo_qty = float(request.form['lboqty'])
            potential_issue_Yes = float(request.form.get('pot_iss'))
            deck_risk_Yes = float(request.form.get('deck_risk'))
            oe_constraint_Yes = float(request.form.get('oe_con'))
            ppap_risk_Yes = float(request.form.get('ppap_risk'))
            stop_auto_buy_Yes = float(request.form.get('stop_autobuy'))
            rev_stop_Yes = float(request.form.get('rev_stop'))
        
            pred_args = [national_inv, lead_time, in_transit_qty, forecast_3_month, sales_1_month, pieces_past_due, perf_6_month_avg, local_bo_qty, potential_issue_Yes, deck_risk_Yes, oe_constraint_Yes, ppap_risk_Yes, stop_auto_buy_Yes, rev_stop_Yes]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            #pred_args_arr = scaler.transform(pred_args_arr)

            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction), 2)
            print(model_prediction)
            if model_prediction == 1.00:
                model_prediction = "Yes"
            else:
                model_prediction = "No"
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction = model_prediction)


if __name__ == "__main__":
    app.run(debug=True)
