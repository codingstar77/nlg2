from flask import Flask, render_template, request
from eval_on_csv import get_prediction_on_csv
import os
app = Flask(__name__)

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
      f = request.files['file']
      path = './uploads/'+f.filename
      f.save(path)
      pred = get_prediction_on_csv(path)
      os.remove(path)
      return render_template('predict.html',prediction = pred)
   




if __name__ == '__main__':
   app.run(debug = True)