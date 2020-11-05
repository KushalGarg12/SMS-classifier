from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


model=pickle.load(open('model.pkl','rb'))
cv=pickle.load(open('transform.pkl','rb'))

app = Flask(__name__)



#cv = CountVectorizer()
@app.route('/')
def home():
	return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
	
	if request.method == 'POST':
		message1= request.form['message']
		data1 = [message1]
		#vect=tfidf.fit(data1)
		vect = cv.transform(data1).toarray()

		my_prediction = model.predict(vect)
		return render_template('result.html',prediction = my_prediction)
		

if __name__ == '__main__':
	app.run(debug=True)