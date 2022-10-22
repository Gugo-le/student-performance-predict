import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

app=Flask(__name__)
## Load the model
regmodel= load_model('studentPerformanceModel.h5', compile=False)
scalar=pickle.load(open('scTransformer.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    print(request.form.values())
    data=[x for x in request.form.values()]
    for i, item in enumerate(data, start=1):
       if i==15:
          t=data[i]
          if t!=0 or t!='0':
            data[i]=np.log(int(t))
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(data,data[15:16])
    output=regmodel.predict(final_input)[0][0]
    return render_template("home.html",prediction_text="The performance of student is {}".format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    if data['absences']>0:
        data['absences']=np.log(data['absences'])
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return json.dumps(str(output[0][0]))
    # return jsonify(output[0][0])

if __name__=="__main__":
    app.run(debug=True)