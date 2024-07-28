import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
app=Flask(__name__)

loaded_pickle_model = pickle.load(open("diabetesdp.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features= [float(x) for x in request.form.values()]
    features=[np.array(float_features)]
    prediction=loaded_pickle_model.predict(features)
    
    return render_template("index.html", prediction_text= "D satge is {}".format(prediction))

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080)