from flask import Flask, request,render_template
import pandas
import numpy as np
import pickle
model = pickle.load(open("model.pkl",'rb'))

# flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predic():
    feature = request.form['feature']
    feature_lst = feature.split(',')
    np_feature  = np.asarray(feature_lst,dtype=np.float32)
    pred = model.predict(np_feature.reshape(1,-1))


    output = ["cancrous" if pred[0] ==1 else "not cancrous"]

    return render_template('index.html',message = output)


# python main
if __name__ == "__main__":
    app.run(debug=True)