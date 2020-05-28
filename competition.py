# Method:
# One method consists of the SVD algorithm using the surprise package
# Another method consists of XGBoost with parameters average_stars of business, user, useful, latitude, longitude, review_count
# Final Result is the weighted average of the above two methods to predict more accurate results

# Error Distribution:
# >=0 and <1: 101472
# >=1 and <2: 33561
# >=2 and <3: 6301
# >=3 and <4: 710
# >=4: 0

# RMSE: 
# 0.9818

# Execution Time: 
# 120sec

import pyspark
import sys
import time
import json
from pyspark import SparkConf, SparkContext
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD, accuracy

initial_time=time.time()
input_path=sys.argv[1]
test_file=sys.argv[2]
output_file=sys.argv[3]
sc= SparkContext()

data = sc.textFile(input_path+"yelp_train.csv")
data=data.map(lambda x: x.split(',')).filter(lambda x: 'user_id' not in x[0]).map(lambda x: (x[0], x[1], x[2]))

testset = sc.textFile(test_file)
testset=testset.map(lambda x: x.split(',')).filter(lambda x: 'user_id' not in x[0]).map(lambda x: (x[0], x[1], x[2]))

bus_dict=sc.textFile(input_path+'business.json').map(lambda x:json.loads(x)).map(lambda x: ((x['business_id']), (x['latitude'], x['longitude'], x['stars'], x['review_count']))).collectAsMap()
users = sc. textFile(input_path+'user.json').map(lambda x:json.loads(x)).map(lambda x: (x['user_id'], (x['average_stars'], x['useful']))).collectAsMap()
# d=review.collect()
# print(d[0])
mat_train, mat_test=[], []
data_labels=[]
testset_labels=[]
#u b lat long avgstarsb revc_b useful_r
for u in data.collect():
	temp=[]
	
	try:
		# print("bussssssss dictttttttttt",bus_dict[u[1]][0])
		# temp.append(u[0])
		# temp.append(u[1])
		temp.append(float(bus_dict[u[1]][0]))
		temp.append(float(bus_dict[u[1]][1]))
		temp.append(float(bus_dict[u[1]][2]))
		# temp.append(float(bus_dict[u[1]][3]))
		temp.append(float(users[u[0]][0]))
		temp.append(float(users[u[0]][1]))
		# temp.append(float(review[(u[0], u[1])]))
		mat_train.append(temp)
		data_labels.append(float(u[2]))
	except KeyError:
		pass	
	
for u in testset.collect():
	temp=[]
	
	try:
		# print("bussssssss dictttttttttt",bus_dict[u[1]][0])
		# temp.append(u[0])
		# temp.append(u[1])
		temp.append(float(bus_dict[u[1]][0]))
		temp.append(float(bus_dict[u[1]][1]))
		temp.append(float(bus_dict[u[1]][2]))
		# temp.append(float(bus_dict[u[1]][3]))
		temp.append(float(users[u[0]][0]))
		temp.append(float(users[u[0]][1]))
		# temp.append(float(review[(u[0], u[1])]))
		mat_test.append(temp)
		# print("tempppppppppp", temp)
		testset_labels.append(float(u[2]))
	except KeyError:
		# temp.append(u[0])
		# temp.append(u[1])
		temp.append(0)
		temp.append(0)
		temp.append(0)
		# temp.append(float(bus_dict[u[1]][3]))
		temp.append(0)
		temp.append(0)
		# temp.append(float(review[(u[0], u[1])]))
		mat_test.append(temp)
		# print("tempppppppppp", temp)
		testset_labels.append(0)
		pass
	
# print("mat testttt", mat_test)
data_features=np.array(mat_train)
# print("Data featuresss", data_features)
testset_features= np.array(mat_test)

xg_reg = xgb.XGBRegressor(objective ='reg:linear', learning_rate = 0.1,
                max_depth = 4, n_estimators=300)

xg_reg.fit(data_features, data_labels)

preds = xg_reg.predict(testset_features)
# print("predictionssss", preds[0])
# rmse = np.sqrt(mean_squared_error(testset_labels, preds))
# print("RMSE: %f" % (rmse))

data_1 = pd.read_csv(input_path+"yelp_train.csv")
testset_1=pd.read_csv(test_file)
data_1 = data_1[['user_id', 'business_id', 'stars']]
testset_1=testset_1[['user_id', 'business_id', 'stars']]
data_1 = data_1.iloc[:, :]

reader = Reader(rating_scale=(1, 5))
data_1 = Dataset.load_from_df(data_1, reader)

trainset = data_1.build_full_trainset()
alg=SVD()
alg.fit(trainset)


numerator,denominator=0,0
one, two, three, four, five=0,0,0,0,0
file= open(output_file, "w")
file.write("user_id, business_id, prediction\n")
for index, p in testset_1.iterrows():
	denominator+=1

	s_stars=alg.predict(p['user_id'], p['business_id'], p['stars'])
	rating= 0.3471853*s_stars.est + 0.6528147*preds[index]
	diff=abs(rating-p['stars'])
	if diff>=0 and diff<1: one+=1
	if diff>=1 and diff<2: two+=1
	if diff>=2 and diff<3: three+=1
	if diff>=3 and diff<4: four+=1
	if diff>=4: five+=1
	numerator+=(rating-p['stars'])**2
	file.write(p['user_id']+', '+p['business_id']+', '+  str(rating)+ "\n")

file.close()
print(one)
print(two)
print(three)
print(four)
print(five)
rmse_final= (numerator/denominator)**0.5
print("Final RMSE", rmse_final)
print("Duration", time.time()-initial_time)
