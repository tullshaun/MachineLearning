import numpy as np
import pickle
import json
import requests
from flask import Flask, jsonify, request,render_template
import pymysql
import uuid
from sklearn.externals import joblib
import pandas as pd
app = Flask(__name__)
connection = pymysql.connect(
    host='localhost',
    user='root',
    password='Nippon2020',
    db='loandb',
)



@app.route('/')

def fpage():
  # loanapp_page = "Loanapp"
   return render_template('Fpage.html')

@app.route('/loanapp')
def loanapp():
   return render_template('Loanapp.html')

@app.route('/dashboard')
def dashboard():
    cursor = connection.cursor()
    sql = "select married, dependents, graduate, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, PropertyOwner,Gender_female, Gender_male from loandb.vw_usrloans;"
    # new_data = json.dumps({'Married': 0, 'Dependents': 5, 'Graduate': 1, 'Self_Employed': 1, 'ApplicantIncome': 5000,'CoapplicantIncome': 1500, 'LoanAmount': 120, 'Loan_Amount_Term': 60, 'Credit_History': 1,'PropertyOwner': 1, 'Gender_Female': 0, 'Gender_Male': 1})

    cursor.execute(sql)
    #cursor.close()
    results =[]
    ndata = []
  
    for row in cursor:
       ndata.append(row)
       #cursor.close()
    sql1 ="select LoanAmount, Loan_Amount_Term  from loandb.vw_loanamnt;"
    cursor.execute(sql1)
    ##for row in cursor:
    ##   results.append(row)
    result = cursor.fetchall()
    cursor.close()
    s = np.array(ndata, dtype=int).reshape(1, 12)
    newdata = s.tolist()
    t = np.array(result, dtype=int)
    results = t.tolist()
    lab = []
    ds = []
    for i in range(len(results)):
        print(i, results[i][1])
        lab.append(results[i][1])
        ds.append(results[i][0])
    mp = 'Naive Bayes Model'
    my_gaumodel_pkl = open('pkl_objects/myGauclassifier.pkl', 'rb')
    my_model = pickle.load(my_gaumodel_pkl)
    res = my_model.predict(newdata)
    pr = my_model.predict_proba(newdata)
    p = pr[0][1] * 100
    pred = round(p, 2)
    if res == 1:
        res = 'Passed'
    else:
        res = 'Failed'
    values = [100 - pred, pred]
    #connection.close()
    return render_template('dashboard.html',results=results,newdata =newdata,my_model =my_model,res=res,ndata=ndata,values=values,pred=pred,lab=lab,ds=ds)

@app.route('/result', methods=['POST','GET'])
def result():

    if request.method == 'POST':
        # Get values from browser
        married = request.form['married']
        dependents = request.form['Dependents']
        graduate = request.form['Graduate']
        self_employed = request.form['Self_Employed']
        applicantIncome = request.form['ApplicantIncome']
        coapplicantIncome = request.form['CoapplicantIncome']
        loanamount = request.form['LoanAmount']
        loan_amount_term = request.form['Loan_Amount_Term']
        credit_history = request.form['Credit_History']
        propertyowner = request.form['PropertyOwner']
        gender_female = request.form['Gender_Female']
        gender_male = request.form['Gender_Male']
        model_predictor = request.form['Model_Predictor']
        #global mm
        #if model_predictor ==1:
        #    mm = my_model_pkl
        #my_model = pickle.load(mm)
        cursor = connection.cursor()
        sql = ("INSERT INTO loandb.loans (Gender, married, dependents, graduate, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, PropertyOwner) values ('male',married, dependents, graduate, self_employed, applicantIncome, coapplicantIncome, loanamount,loan_amount_term, credit_history, propertyowner)")
        cursor.execute(sql)
        connection.commit()
        cursor.close()
        #connection.close()
        s = np.array([married, dependents, graduate, self_employed, applicantIncome, coapplicantIncome, loanamount,loan_amount_term, credit_history, propertyowner, gender_female, gender_male], dtype=int).reshape(1, 12)
        testData = s.tolist()
        uid = uuid.uuid4()
        if int(model_predictor) == 1:
            mp = 'KNNeighbours Model'
            #my_model = joblib.load('pkl_objects/myKnnlassifier.pkl')
            with open('pkl_objects/myKnnlassifier.pkl', 'rb') as my_model_pkl:
                my_model = pickle.load(my_model_pkl)
            #mm =my_model_pkl
        if int(model_predictor) == 2:
            mp = 'Naive Bayes Model'
            with open('pkl_objects/myGauclassifier.pkl', 'rb') as my_gaumodel_pkl:
                my_model = pickle.load(my_gaumodel_pkl)
        if int(model_predictor) == 3:
            mp = 'Neural Network Model'
            with open('pkl_objects/mynnlassifier.pkl', 'rb') as my_nnmodel_pkl:
                my_model = pickle.load(my_nnmodel_pkl)

        res = my_model.predict(testData)
        pr = my_model.predict_proba(testData)
        p =pr[0][1] * 100
        pred =round(p,2)
        if res ==1:
            res ='Passed'
        else:
            res='Failed'
        values =[100-pred,pred]
        result = request.form
        return render_template("result.html", result=result, res=res,testData=testData,pred=pred,pr=pr,model_predictor = model_predictor,mp=mp,values=values,uid=uid)








if __name__ == '__main__':
     #if model_predictor ==1:
     #    print('yes')
     #my_model_pkl = open('pkl_objects/myClassifier.pkl', 'rb')
     #my_model = pickle.load(my_model_pkl)
     #my_nnmodel_pkl = open('pkl_objects/mynnlassifier.pkl', 'rb')
     ##my_dtmodel = pickle.load(my_dtmodel_pkl)
     ##my_gaumodel_pkl = open('pkl_objects/myGauclassifier.pkl', 'rb')



    
     app.run(debug=True)
