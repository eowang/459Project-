import pandas as pd
import sys
import matplotlib.pyplot as plt 
import numpy as np
import statistics
import time

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss

#
#Libraries for stratified sampling, to run on smaller sample subset of rows from each feature
#
from sklearn.model_selection import StratifiedShuffleSplit
#Library for mean absolute error measurement
from sklearn.metrics import mean_absolute_error

def ndis(row, DF2):
 
    latitude,longitude=row['latitude'],row['longitude']
   
    DF2['latitude']= DF2['latitude'].astype(float)
    DF2['longitude']= DF2['longitude'].astype(float)
    

    DF2['DIS']=(DF2.latitude.sub(latitude).pow(2).add(DF2.longitude.sub(longitude).pow(2))).pow(.5)   

    DF2.to_csv("sdfadsf.csv", index=False)

    return DF2.DIS.min()

def euclidDT(row):
    dtLat = 40.7580
    dtLong = -73.9855
    
    latitude,longitude =row['latitude'],row['longitude']

    return pow((pow((dtLat-latitude),2) + pow((dtLong-longitude),2)),.5) 

def Nearest_Subway(data):

    subwayData= pd.read_csv("subwayData.csv")

    subwayData['longitude']= subwayData['the_geom'].apply(lambda st: st[st.find("(")+1:st.find("4")])
    subwayData['latitude']= subwayData['the_geom'].apply(lambda st: st[st.find(" 4")+1:st.find(")")])
    
    data['min_euc_dist']=data.apply(lambda x : ndis(x, subwayData), axis=1)

    
def Distance_Downtown(data): 
    data['dist_dt']=data.apply(lambda x : euclidDT(x), axis=1)

#Recursive feature elimination
def recursiveFeature(X,y):
    # feature extraction
    X_scaled = preprocessing.scale(X)
    model = LogisticRegression(solver='lbfgs',max_iter=2000)
    rfe = RFE(model, 20)
    fit = rfe.fit(X_scaled, y)
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)

def Feature_Selection(data):
    data = data.drop(data.columns[[2,3,4,5,6,10,11,13,16]], axis=1)
    
    cols = list(data)
    cols.insert(-1, cols.pop(cols.index('interest_level')))
    cols.insert(-1, cols.pop(cols.index('dist_dt')))
    data = data[cols]
    data['longitude']= data['longitude'].abs()
    data['Roof Deck']= data['Roof Deck'] | data['Roof-deck']
    data['High Ceiling']= data['High Ceiling'] | data['High Ceilings'] |data['HIGH CEILINGS']
    data['Renovated'] = data['Renovated'] | data['Newly renovated']
    data['Live In Super'] = data['Live In Super'] | data['LIVE IN SUPER']
    data['Gym/Fitness'] = data['Gym/Fitness'] | data['Fitness Center']
    data.Pool = data.Pool | data['Swimming Pool']
    data ['Hardwood'] = data.Hardwood | data.HARDWOOD | data['Hardwood Floors']
    data['Outdoor Space'] = data['Outdoor Space'] | data['Common Outdoor Space'] | data['Private Outdoor Space'] | data['PublicOutdoor']
    data['Elevator'] = data['elevator'] | data['Elevator'] 
    data['LAUNDRY']= data['Laundry in Building'] | data['Laundry in Unit'] | data['Laundry In Building'] | data['Laundry Room'] | data['Laundry In Unit'] | data['LAUNDRY'] | data['On-site laundry'] | data['On-site Laundry'] | data['Washer/Dryer']
    data['Pre-War'] = data['Pre-War'] | data['prewar'] | data['Prewar']
    data['Pets on approval']= data['Pets on approval'] | data['Dogs Allowed'] | data['Cats Allowed']
    data.dishwasher = data.Dishwasher | data.dishwasher
    data.Garage = data.Garage | data['On-site Garage']
    data.drop(['Dogs Allowed','Cats Allowed','LIVE IN SUPER','Newly renovated','Washer/Dryer','HIGH CEILINGS','High Ceilings','Swimming Pool','Roof-deck','On-site Garage','Common Outdoor Space','Private Outdoor Space', 'PublicOutdoor','elevator','HARDWOOD', 'Hardwood Floors', 'Laundry in Building','Laundry in Unit','Laundry In Building','Laundry Room','Laundry In Unit', 'On-site laundry', 'On-site Laundry','prewar','Prewar','Dishwasher'], axis = 1, inplace=True) 
    
  

    # X = data.drop['interest_level']  #independent columns
    # y = data['interest_level']   #target column i.e price range
    # data.to_csv('test_data.csv', index=False)
  
    # recursiveFeature(X,y)

def Decision_Tree(X,y):
# 
#Current production code for Decision Tree Classifier
#
    kf = KFold(n_splits = 5)

    scores = []

    KFold(n_splits=5, random_state=None, shuffle=False)

    dec_Tree_class = DecisionTreeClassifier(criterion='entropy', max_depth=5)

    for train_index, test_index in kf.split(X):
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
        dec_Tree_class = dec_Tree_class.fit(X_train, y_train)
        pred_prob = dec_Tree_class.predict_proba(X_valid)
        
        scores.append(log_loss(y_valid, pred_prob))
        
        #Model validation on X_train to test if model is overfitting.
        # Predicted proba(X_train) provides predicted prob on X_train so we can input it into log_loss function.
        xtrain_error_prob= dec_Tree_class.predict_proba(X_train)
        
        log_overfit_scores.append(log_loss(X_train, xtrain_error_prob))
        acc_overfit_scores.append(dec_Tree_class.score(X_train, y_train))

        avgOverfit_log_score = statistics.mean(log_overfit_scores)
        avgOverfit_acc_score = statistics.mean(acc_overfit_scores)

    avg_score = statistics.mean(scores)
    print("Scores are:", scores)
    print("Average score is:", avg_score) 
    
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


    

# KFOLD IMPLEMENTATION for Decision Tree Classifier
    # col_names = ['interest_level',	'hour_created',	'numeric_interest_level',	'word_count',	'Elevator',	'Hardwood Floors',	'Cats Allowed',	'Dogs Allowed',	'Doorman',	'Dishwasher',	'No Fee	Laundry in Building	Fitness Center',	'Pre-War',	'Laundry in Unit',	'Roof Deck',	'Outdoor Space',	'Dining Room',	'High Speed Internet',	'Balcony',	'Swimming Pool',	'Laundry In Building',	'New Construction',	'Terrace',	'Exclusive	Loft',	'Garden/Patio',	'Wheelchair Access',	'Common Outdoor Space',	'HARDWOOD',	'Fireplace',	'SIMPLEX',	'prewar',	'LOWRISE',	'Garage	Laundry Room',	'Reduced Fee',	'Laundry In Unit',	'Furnished',	'Multi-Level',	'Private Outdoor Space',	'Prewar	PublicOutdoor',	'Parking Space',	'Roof-deck',	'dishwasher',	'High Ceilings',	'elevator',	'Renovated',	'Pool',	'LAUNDRY',	'Green Building',	'HIGH CEILINGS',	'LIVE IN SUPER',	'High Ceiling',	'Washer in Unit',	'Dryer in Unit',	'Storage',	'Stainless Steel Appliances',	'On-site laundry',	'Concierge',	'Newly renovated',	'On-site Laundry',	'Hardwood',	'Light',	'Live In Super',	'On-site Garage', 'Washer/Dryer',	'Granite Kitchen',	'Gym/Fitness',	'Pets on approval',	'Marble', 'Bath',	'Walk in Closet(s)',	'Subway']
    # col_names = ['longitude', 'latitude', 'bedrooms', 'bathrooms']   
    # data.head(5000)

    # X = data[col_names].sample(n=300)

    # y = data.interest_level.sample(n=300)


def Decision_tree_kaggle(test_data, X, y, listing_id):
    tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    tree_model.fit(X, y)

    predicted = tree_model.predict_proba(test_data)

    col_names = tree_model.classes_
    df_predict_proba = pd.DataFrame(predicted, columns= col_names)

    df_predict_proba['listing_id'] = listing_id.astype(int)
    col_names = ['listing_id', 'high', 'medium', 'low']

    df_predict_proba = df_predict_proba[col_names]
    df_predict_proba= df_predict_proba.dropna()
    df_predict_proba.to_csv('kaggle_submission_TREE.csv', index=False)

def Logistic_Regression_score(X, y):
   
    # X= data[['bathrooms','bedrooms','latitude','longitude','price','hour_created','word_count','numeric_interest_level']]
    # X = data.drop(['numeric_interest_level','interest_level', 'photos', 'description', 'features', 'listing_id', 'display_address', 'building_id', 'created', 'street_address', 'manager_id'], axis=1)
    # y = data['numeric_interest_level']
    log_scores = []
    acc_scores = []
    
    kf = KFold(n_splits = 5)

    KFold(n_splits=2, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X):
        #print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
        # print(y_train)
        logistic_model = LogisticRegression()
        logistic_model.fit(X_train, y_train)

        predicted_prob = logistic_model.predict_proba(X_valid)
        log_scores.append(log_loss(y_valid, predicted_prob))
        acc_scores.append(log_loss(y_valid, predicted_prob))

    avg_log_score = statistics.mean(log_scores)
    avg_acc_score = statistics.mean(acc_scores)
    print("\n\nLogistic regression score:")
    print("Average log loss score:", avg_log_score)
    print("Average accuracy score is:", avg_acc_score)

def Logistic_Regression_kaggle(test_data, X, y, listing_id):
    logistic_model = LogisticRegression()
    logistic_model.fit(X, y)
    print("classes are:", logistic_model.classes_)
    predicted = logistic_model.predict_proba(test_data)

    print(len(predicted))
    col_names = ['high', 'low', 'medium']
    df_predict_proba = pd.DataFrame(predicted, columns= col_names)

    print(df_predict_proba.shape)
    df_predict_proba['listing_id'] = listing_id.astype(int)
    col_names = ['listing_id', 'high', 'medium', 'low']

    df_predict_proba = df_predict_proba[col_names]
    print(df_predict_proba.shape)
    df_predict_proba= df_predict_proba.dropna()
    df_predict_proba.to_csv('kaggle_submission_LR.csv', index=False)

def SVM_score(X, y):
    # X = data.drop(['interest_level', 'photos', 'description', 'features', 'listing_id', 'display_address', 'building_id', 'created', 'street_address', 'manager_id'], axis=1)
    # y = data['interest_level']
    start_time = time.time()
    log_scores = []
    acc_scores = []
    kf = KFold(n_splits = 5)
    
    KFold(n_splits=2, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X):
        #print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

        svm_model = SVC(C=1, max_iter=100, kernel='rbf', probability=True, gamma='auto')
        svm_model.fit(X_train, y_train)

        predicted_prob = svm_model.predict_proba(X_valid)
        log_scores.append(log_loss(y_valid, predicted_prob))
        acc_scores.append(svm_model.score(X_valid, y_valid))

    avg_log_score = statistics.mean(log_scores)
    avg_acc_score = statistics.mean(acc_scores)
    print("\n\nSVM scores: ")
    print("Average log loss score:", avg_log_score)
    print("Average accuracy score:", avg_acc_score)

    print("run time of SVM", time.time() - start_time)

def SVM_kaggle(test_data, X, y, listing_id):
    start_time = time.time()
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X, y)

    predicted = svm_model.predict_proba(test_data)

    col_names = svm_model.classes_
    df_predict_proba = pd.DataFrame(predicted, columns= col_names)

    df_predict_proba['listing_id'] = listing_id.astype(int)
    col_names = ['listing_id', 'high', 'medium', 'low']

    df_predict_proba = df_predict_proba[col_names]
    df_predict_proba= df_predict_proba.dropna()
    df_predict_proba.to_csv('kaggle_submission_SVM.csv', index=False)

    print("run time of SVM", time.time() - start_time)

if __name__ == '__main__':
    #
    #Run with python3 milestone2.py data_after_M1.csv
    #
    data = pd.read_csv(sys.argv[1])
    data.sample(n=5000)
    print(data.shape)

    #provide the most signifcantly correlated features to 'interest_level' feature.
    X = data[['bathrooms','bedrooms','latitude','longitude','price','hour_created','word_count','Doorman','No Fee','Pre-War','Dining Room', 'Balcony','SIMPLEX','LOWRISE','Garage','Reduced Fee','Furnished','LAUNDRY','Hardwood','Subway']]
    y = data['interest_level']
    #
    #Stratified sampling of our Y dependent class, this will sample each stratum(low,med,high) 
    #in our interest_level feature equally.
    #

    # y_data = StratifiedShuffleSplit(n_splits=3, test_size = 300, random_state=2)
    # y_data.get_n_splits(X,y)

    # for train_index, test_index in y_data.split(X, y):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]

    # print(y_data)
    # data = pd.read_json(sys.argv[1])
    # data = data.sample(n=140)
    # X= data[['bathrooms','bedrooms','latitude','longitude','price','hour_created','word_count','Doorman','No Fee','Pre-War','Dining Room', 'Balcony','SIMPLEX','LOWRISE','Garage','Reduced Fee','Furnished','LAUNDRY','Hardwood','Subway']]
    # y = data['interest_level']
    #
    #Preprocessing
    #
    # Nearest_Subway(data)
    # Distance_Downtown(data)
    # Feature_Selection(data)
    #
    #Classifiers
    #
    Decision_Tree(X,y)
    # SVM_score(X, y)
    # Logistic_Regression_score(X, y)
    # Nearest_Subway(data)