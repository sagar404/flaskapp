from flask import Flask,request, jsonify, render_template
import pickle
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
model1 = pickle.load(open('logreg.pkl','rb'))
model2 = pickle.load(open('nb.pkl','rb'))
model3 = pickle.load(open('rf.pkl','rb'))


vect = CountVectorizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if (request.method=='POST'):
        message = request.form['message']
        operation=request.form['operation']
        
        if(operation=='Logistic Regression'):
            predic1 = model1.predict(vect.transform([message]))[0]
            output = 'LogiticRegression = {}'.format(predic1)
        
        if(operation=='Naive Baise'):
            predic2 = model2.predict(vect.transform([message]))[0]
            output = 'Naive Baise = {}'.format(predic2)
        
        if(operation=='Random Forest'):
            predic3 = model3.predict(vect.transform([message]))[0]
            output = 'Random Forest = {}'.format(predic3)
        
        
    
    return render_template('index.html', result ='Output is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
