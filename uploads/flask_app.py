from flask import Flask, render_template, request,redirect,session
from keras.models import load_model
from eval_on_csv import get_prediction_on_csv,softmax_over_time
import pickle

import os
app = Flask(__name__)
app.secret_key = "98er72ehf7e77237h82hjdnsjdhjdnshd"

@app.route('/')
def index():
      if not session.get('username'):
            return render_template('index.html',message = None)
      else:
            return redirect('/predict')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if not session.get('username'):
      return render_template('index.html',message = "Please Login First..")


    if request.method == 'POST':
      f = request.files['file']
      path = './uploads/'+f.filename
      f.save(path)
      pred = get_prediction_on_csv(path)
      os.remove(path)
      return render_template('predict.html',prediction = pred)
    elif request.method == 'GET':
          return render_template('predict.html',prediction = None)


   
@app.route('/login', methods = ['POST'])
def validate():
      valid_users = ['kd','hemant','parmeshwar','sourabh']
      valid_passwords = {user: user+"123" for user in valid_users}
      user = request.form.get("username")
      password = request.form.get("password") 
      if user and password and valid_passwords.get(user) == password:
            session['username'] = user
            return redirect('/predict')
      else:
            return render_template('index.html',message="Invalid Username or Password.")


@app.route('/logout', methods = ['GET'])
def logout():
      if  session.get("username"):
            session['username'] = None
      return render_template('index.html',message="Logged Out Successfully..")
      

if __name__ == '__main__':
    
    app.run(debug = True)


