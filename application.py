# importing necessary libraries
import pickle
# from joblib import dump, load
from flask import Flask,render_template,request
import pandas as pd
import numpy as np

#Create flask app
app= Flask(__name__)

#load the model
classifier = pickle.load(open('classifier.pkl','rb'))
# classifier = load('classifier.joblib') 

@app.route("/")
def home():
    return render_template('codegnanApp.html')

@app.route("/predict", methods=["POST"])
def predict():
    input_data = [ x for x in request.form.values()]
    input_data = [np.array(input_data)]

    print(input_data)
#     print(input_data.pop())
    print(input_data)
    prediction=classifier.predict(pd.DataFrame(input_data, columns=['Income','Kidhome','Teenhome','Age','Partner','Education_Level']))
    print(prediction)
    pred_1 = 0
    if prediction == 0:
            pred_1 = 'cluster 1'

    elif prediction == 1:
            pred_1 = 'cluster 2'

    elif prediction == 2:
            pred_1 = 'cluster 3'

    elif prediction == 3:
            pred_1 = 'cluster 4'

    return render_template("codegnanApp.html", prediction_text = "The customer belongs to {}".format(pred_1))

if __name__ == "__main__":
    app.run(debug=True)
    


