# Yelp-Recommendation-System
Using Spark RDD

Designed a recommendation system for the Yelp dataset using weighted average of SVD algorithm from surpris package and XGBoost algorithm with average_stars of business, user, useful, latitude, longitude, review_count as parameters in order to achieve lower RMSE and efficient time.

Error Distribution:
 >=0 and <1: 101472<br>
 >=1 and <2: 33561<br>
 >=2 and <3: 6301<br>
 >=3 and <4: 710<br>
 >=4: 0<br>

RMSE: 
0.9818

Execution Time: 
120sec
