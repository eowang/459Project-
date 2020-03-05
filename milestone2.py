import pandas as pd
import sys
import matplotlib.pyplot as plt 
import numpy as np
import statistics

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

#
#Libraries for stratified sampling, to run on smaller sample subset of rows from each feature
#

from sklearn.model_selection import StratifiedShuffleSplit


# DF2= pd.read_csv("subwayData.csv")
def Min_Euclidean_Dist(row):
    data2['distance']= data2.apply(lambda x: np.square())
    return np.linalg.norm(data1[cols].values - data2[cols].values,
                   axis=1)

def ndis(row, DF2):
    try:
        latitude,longitude=row['latitude'],row['longitude']
        # print(DF2.latitude)
        # DF2['DIS']=(DF2.latitude-latitude)*(DF2.latitude-latitude)+(DF2.longitude-longitude)*(DF2.longitude-longitude)
        DF2['DIS']=DF2[['latitude', 'longitude']].sub(np.array([latitude,longitude])).pow(2).sum(1).pow(0.5)

        print(DF2.DIS)
        DF2.to_csv("sdfadsf", index=false)
        # temp=DF2.ix[DF2.DIS.min()]
        # return temp[2]  
        return DF2.DIS.min()
    except:
        pass        
    

def Nearest_Subway(data):
    subwayData= pd.read_csv("subwayData.csv")
    # subwayData['coordinates'] = subwayData['the_geom'].str.split(' (').str[0]


    subwayData['longitude']= subwayData['the_geom'].apply(lambda st: st[st.find("(")+1:st.find("4")])
    subwayData['latitude']= subwayData['the_geom'].apply(lambda st: st[st.find(" 4")+1:st.find(")")])
    
    data['min_euc_dist']=data.apply(lambda x : ndis(x, subwayData), axis=1)
    
    data.to_csv('testtest.csv', index=False)
    # distance_array = np.sum((data[xy].values - subwayData[xy].values)**2, axis=1)
    # test = distance_array.argmin()
    # print(test)
    # distanceArray= Min_Euclidean_Dist(data,subwayData)
    
    # data.apply(lambda x: Min_Euclidean_Dist())


# def Feature_Selection(data):

def Decision_Tree(X,y):
#    
# Attempt at writing Decision Tree with cross_validation.train_test_split
# 
   
    #'features'
    #col_names = ['interest_level',	'hour_created',	'numeric_interest_level',	'word_count',	'Elevator',	'Hardwood Floors',	'Cats Allowed',	'Dogs Allowed',	'Doorman',	'Dishwasher',	'No Fee	Laundry in Building	Fitness Center',	'Pre-War',	'Laundry in Unit',	'Roof Deck',	'Outdoor Space',	'Dining Room',	'High Speed Internet',	'Balcony',	'Swimming Pool',	'Laundry In Building',	'New Construction',	'Terrace',	'Exclusive	Loft',	'Garden/Patio',	'Wheelchair Access',	'Common Outdoor Space',	'HARDWOOD',	'Fireplace',	'SIMPLEX',	'prewar',	'LOWRISE',	'Garage	Laundry Room',	'Reduced Fee',	'Laundry In Unit',	'Furnished',	'Multi-Level',	'Private Outdoor Space',	'Prewar	PublicOutdoor',	'Parking Space',	'Roof-deck',	'dishwasher',	'High Ceilings',	'elevator',	'Renovated',	'Pool',	'LAUNDRY',	'Green Building',	'HIGH CEILINGS',	'LIVE IN SUPER',	'High Ceiling',	'Washer in Unit',	'Dryer in Unit',	'Storage',	'Stainless Steel Appliances',	'On-site laundry',	'Concierge',	'Newly renovated',	'On-site Laundry',	'Hardwood',	'Light',	'Live In Super',	'On-site Garage', 'Washer/Dryer',	'Granite Kitchen',	'Gym/Fitness',	'Pets on approval',	'Marble', 'Bath',	'Walk in Closet(s)',	'Subway']
    # col_names = ['longitude', 'latitude', 'bedrooms', 'bathrooms']   
    # data.head(5000)

    # X = data[col_names].sample(n=300)

    # y = data.numeric_interest_level.sample(n=300)

    # X_train, X_test, y_train, y_test, = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)

    # dec_Tree_class = DecisionTreeClassifier()

    # dec_Tree_class = dec_Tree_class.fit(X_train, y_train)

    # y_pred = dec_Tree_class.predict(X_test)

    # scores = cross_val_score(dec_Tree_class, X, y, cv = 5)

    # print("Accuracy:", scores.mean(), scores.std() * 2)
    # print("Accuracy:", dec_Tree_class.accuracy.score(y_test, y_pred))


    

# KFOLD IMPLEMENTATION
    # col_names = ['interest_level',	'hour_created',	'numeric_interest_level',	'word_count',	'Elevator',	'Hardwood Floors',	'Cats Allowed',	'Dogs Allowed',	'Doorman',	'Dishwasher',	'No Fee	Laundry in Building	Fitness Center',	'Pre-War',	'Laundry in Unit',	'Roof Deck',	'Outdoor Space',	'Dining Room',	'High Speed Internet',	'Balcony',	'Swimming Pool',	'Laundry In Building',	'New Construction',	'Terrace',	'Exclusive	Loft',	'Garden/Patio',	'Wheelchair Access',	'Common Outdoor Space',	'HARDWOOD',	'Fireplace',	'SIMPLEX',	'prewar',	'LOWRISE',	'Garage	Laundry Room',	'Reduced Fee',	'Laundry In Unit',	'Furnished',	'Multi-Level',	'Private Outdoor Space',	'Prewar	PublicOutdoor',	'Parking Space',	'Roof-deck',	'dishwasher',	'High Ceilings',	'elevator',	'Renovated',	'Pool',	'LAUNDRY',	'Green Building',	'HIGH CEILINGS',	'LIVE IN SUPER',	'High Ceiling',	'Washer in Unit',	'Dryer in Unit',	'Storage',	'Stainless Steel Appliances',	'On-site laundry',	'Concierge',	'Newly renovated',	'On-site Laundry',	'Hardwood',	'Light',	'Live In Super',	'On-site Garage', 'Washer/Dryer',	'Granite Kitchen',	'Gym/Fitness',	'Pets on approval',	'Marble', 'Bath',	'Walk in Closet(s)',	'Subway']
    # col_names = ['longitude', 'latitude', 'bedrooms', 'bathrooms']   
    # data.head(5000)

    # X = data[col_names].sample(n=300)

    # y = data.interest_level.sample(n=300)
# 
#
#
    kf = KFold(n_splits = 4)

    scores = []

    KFold(n_splits=4, random_state=None, shuffle=False)

    dec_Tree_class = DecisionTreeClassifier(criterion='entropy')

    for train_index, test_index in kf.split(X):
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

    dec_Tree_class = dec_Tree_class.fit(X_train, y_train)
    
    pred_prob = dec_Tree_class.predict_proba(X_valid)
    scores.append(log_loss(y_valid, pred_prob))

    avg_score = statistics.mean(scores)
    print("Scores are:", scores)
    print("Average score is:", avg_score) 

# 
# 
# 
    # col_names = ['interest_level',	'hour_created',	'numeric_interest_level',	'word_count',	'Elevator',	'Hardwood Floors',	'Cats Allowed',	'Dogs Allowed',	'Doorman',	'Dishwasher',	'No Fee	Laundry in Building	Fitness Center',	'Pre-War',	'Laundry in Unit',	'Roof Deck',	'Outdoor Space',	'Dining Room',	'High Speed Internet',	'Balcony',	'Swimming Pool',	'Laundry In Building',	'New Construction',	'Terrace',	'Exclusive	Loft',	'Garden/Patio',	'Wheelchair Access',	'Common Outdoor Space',	'HARDWOOD',	'Fireplace',	'SIMPLEX',	'prewar',	'LOWRISE',	'Garage	Laundry Room',	'Reduced Fee',	'Laundry In Unit',	'Furnished',	'Multi-Level',	'Private Outdoor Space',	'Prewar	PublicOutdoor',	'Parking Space',	'Roof-deck',	'dishwasher',	'High Ceilings',	'elevator',	'Renovated',	'Pool',	'LAUNDRY',	'Green Building',	'HIGH CEILINGS',	'LIVE IN SUPER',	'High Ceiling',	'Washer in Unit',	'Dryer in Unit',	'Storage',	'Stainless Steel Appliances',	'On-site laundry',	'Concierge',	'Newly renovated',	'On-site Laundry',	'Hardwood',	'Light',	'Live In Super',	'On-site Garage', 'Washer/Dryer',	'Granite Kitchen',	'Gym/Fitness',	'Pets on approval',	'Marble', 'Bath',	'Walk in Closet(s)',	'Subway']
    # #'features'
    # X = data[col_names].iloc[:300]

    # y = data.numeric_interest_level.iloc[:300]

    # X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.3, random_state=1)

    # dec_Tree_class = DecisionTreeClassifier(criterion='entropy')

    # dec_Tree_class = dec_Tree_class.fit(X_train, y_train)

    # y_pred = dec_Tree_class.predict(X_test)

    # print("Accuracy:", metrics.accuracy.score(y_test, y_pred))



# def Logistic_Regression(data):

# def SVM(data):

if __name__ == '__main__':
    #
    #Run with python3 milestone2.py data_after_M1.csv
    #
    data = pd.read_csv(sys.argv[1])

    #provide the most signifcantly correlated features to 'interest_level' feature.
    X = data[['bathrooms','bedrooms','latitude','longitude','price','hour_created','word_count','Doorman','No Fee','Pre-War','Dining Room', 'Balcony','SIMPLEX','LOWRISE','Garage','Reduced Fee','Furnished','LAUNDRY','Hardwood','Subway']]
    y = data['interest_level']
    #
    #Stratified sampling of our Y dependent class, this will sample each stratum(low,med,high) 
    #in our interest_level feature equally.
    #
    y_data = StratifiedShuffleSplit(n_splits=3, test_size = 5000, random_state=2)
    y_data.get_n_splits(X,y)

    print(y_data)

    # print(data.head(n=300))
    # print(type(data.iloc[:300]))
    # print(type(data.numeric_interest_level[1]))
	# data = pd.read_json(sys.argv[1])
	# Feature_Selection(data)
    # Decision_Tree(X,y)
	# Logistic_Regression(data)
	SVM(X,y)
    # Nearest_Subway(data)
