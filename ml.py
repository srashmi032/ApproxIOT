#import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
from  matplotlib import pyplot  as plt
from matplotlib import style
import datetime
import pandas
import time
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn import utils
from sklearn.linear_model import Lasso

import pickle

def stock():
	st=time.time()
	style.use('ggplot')

	#df = quandl.get("WIKI/GOOGL")
	names =['VendorID','pickup_datetime','dropoff_datetime','passenger_count','trip_distance','pickup_longitude',
	'pickup_latitude','RateCodeID','store_and_fwd_flag','dropoff_longitude','dropoff_latitude','payment_type',
	'fare_amount','extra','mta_tax','tip_amount','tolls_amount','improvement_surcharge','total_amount','pickup_zip',
	'pickup_borough','pickup_neighborhood','dropoff_zip','dropoff_borough','dropoff_neighborhood']
	
	dataset = pandas.read_csv('sample_data.csv',names=names)
	df=pandas.DataFrame(dataset)
	df=df.drop(df.index[0])
	print (df)
	#df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
	#df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
	#df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

	df = df[['passenger_count','trip_distance','total_amount','pickup_zip','dropoff_zip']]

	#forecast_col = 'Adj. Close'
	#df.fillna(value=-99999, inplace=True)
	#forecast_out = int(math.ceil(0.01 * len(df)))
	#df['label'] = df[forecast_col].shift(-forecast_out)


	#df['Avg']=pandas.Series(np.random.randn(dataset['Adj. Close'].count()), index=df.index)
	#df['execution_time']=pandas.Series(np.random.randn(dataset['Adj. Close'].count()), index=df.index)
	#df['Moving Avg']=pandas.Series(np.random.randn(dataset['Adj. Close'].count()), index=df.index)

	
	#df['pickup_datetime'] = df['pickup_datetime'].astype('float64') 
	#df['dropoff_datetime'] = df['dropoff_datetime'].astype('float64') 
	print (df)

	ed=time.time()

	print ((ed-st)*1000)

	y= df['total_amount']

	df=df.drop('total_amount',axis=1)
	X=df
	#X = preprocessing.scale(X)
	#X_lately = X[-forecast_out:]
	#X = X[:-forecast_out]

	#df.dropna(inplace=True)

	

	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

	print ("\nX_train:\n")
	print(X_train)
	#print (X_train.shape)
	print ("\nX_test:\n")
	print(X_test)
	#print (X_test.shape)
	print (y_test)
	print("LinearRegression:")


	clf = LinearRegression(n_jobs=-1)
	clf.fit(X_train, y_train)
	confidence = clf.score(X_test, y_test)

	print (confidence)

	prediction=clf.predict(X_test)

	print(prediction)
	#pyplot.scatter(X_test, y_test)
	#plt.scatter(X_test[:,1],prediction)
	#plt.plot(X_test,prediction)
	#plt.show()
	#plt.plot(y_test)
	#plt.plot(prediction)
	#plt.show()

	filename = 'model.sav'
	pickle.dump(clf, open(filename, 'wb'))

	

if __name__=="__main__":
	stock()