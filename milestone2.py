import pandas as pd
import sys
import matplotlib.pyplot as plt 
import numpy as np
import statistics
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, f1_score, confusion_matrix


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
    # cols.insert(-1, cols.pop(cols.index('dist_dt')))
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

def Decision_Tree(X, y):
    log_scores = []
    acc_scores = []

    kf = KFold(n_splits = 5)

    dec_Tree_class = DecisionTreeClassifier(criterion='entropy')

    for train_index, test_index in kf.split(X):
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

        dec_Tree_class = dec_Tree_class.fit(X_train, y_train)
        
        pred_prob = dec_Tree_class.predict_proba(X_valid)
        log_scores.append(log_loss(y_valid, pred_prob))
        acc_scores.append(dec_Tree_class.score(X_valid, y_valid))

    avg_log_score = statistics.mean(log_scores)
    avg_acc_score = statistics.mean(acc_scores)
    print("\n\nDecision tree scores:")
    print("Average log loss score:", avg_log_score)
    print("Average accuracy score:", avg_acc_score)


def Decision_tree_kaggle(test_data, X, y, listing_id):
    tree_model = DecisionTreeClassifier(criterion='entropy')
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
    log_scores = []
    acc_scores = []
    kf = KFold(n_splits = 5)

    KFold(n_splits=2, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X):
        #print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
        print(y_train)
        logistic_model = LogisticRegression()
        logistic_model.fit(X_train, y_train)

        pred_prob = logistic_model.predict_proba(X_valid)
        log_scores.append(log_loss(y_valid, pred_prob))
        acc_scores.append(logistic_model.score(X_valid, y_valid))

    avg_log_score = statistics.mean(log_scores)
    avg_acc_score = statistics.mean(acc_scores)
    print("\n\nLogistic regression score:")
    print("Average log loss score:", avg_log_score)
    print("Average accuracy score:", avg_acc_score)

    # predicted = logistic_model.predict_proba(test_data)

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
    log_scores = []
    acc_scores = []
    kf = KFold(n_splits = 5)

    # KFold(n_splits=2, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X):
        #print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

        svm_model = make_pipeline(
            SVC(kernel='linear', probability=True, max_iter=10)

            )
        svm_model.fit(X_train, y_train)

        pred_prob = svm_model.predict_proba(X_valid)
        log_scores.append(log_loss(y_valid, pred_prob))
        acc_scores.append(svm_model.score(X_valid, y_valid))
        
    avg_log_score = statistics.mean(log_scores)
    avg_acc_score = statistics.mean(acc_scores)
    print("\n\nSVM scores: ")
    print("Average log loss score:", avg_log_score)
    print("Average accuracy score:", avg_acc_score)

    
def SVM_kaggle(test_data, X, y, listing_id):
    svm_model = SVC(kernel='linear', probability=True, max_iter=10)
    svm_model.fit(X, y)

    predicted = svm_model.predict_proba(test_data)

    col_names = svm_model.classes_
    df_predict_proba = pd.DataFrame(predicted, columns= col_names)

    df_predict_proba['listing_id'] = listing_id.astype(int)
    col_names = ['listing_id', 'high', 'medium', 'low']

    df_predict_proba = df_predict_proba[col_names]
    df_predict_proba= df_predict_proba.dropna()
    df_predict_proba.to_csv('kaggle_submission_SVM.csv', index=False)
    

if __name__ == '__main__':
    data = pd.read_csv('data_after_M1.csv')
    # data = pd.read_json(sys.argv[1])
    test_data = pd.read_csv('test_raw_data.csv')
    # test_data = pd.read_json(sys.argv[1])
    test_data.to_csv('test.csv', index = False)
    listing_id = test_data['listing_id']
    print("Listing Id shape:", listing_id.shape)
    # test_data =data.drop(data.columns[[0,3,4,5,6,7,9,11,12,14]])
    test_data = test_data[['bathrooms','bedrooms','latitude','longitude','price','hour_created','word_count','Doorman','No Fee','Pre-War','Dining Room', 'Balcony','SIMPLEX','LOWRISE','Garage','Reduced Fee','Furnished','LAUNDRY','Hardwood','dist_dt']]
    print("After dropping:", test_data.shape)
    data = data.sample(n=140)
    # X= data[['bathrooms','bedrooms','latitude','longitude','price','hour_created','word_count','Doorman','No Fee','Pre-War','Dining Room', 'Balcony','SIMPLEX','LOWRISE','Garage','Reduced Fee','Furnished','LAUNDRY','Hardwood','dist_dt']]
    y = data['interest_level']
    # Nearest_Subway(data)
    Distance_Downtown(data)
    X= data[['bathrooms','bedrooms','latitude','longitude','price','hour_created','word_count','Doorman','No Fee','Pre-War','Dining Room', 'Balcony','SIMPLEX','LOWRISE','Garage','Reduced Fee','Furnished','LAUNDRY','Hardwood','dist_dt']]

    Feature_Selection(data)
    # Decision_tree_kaggle(test_data, X, y, listing_id)
    # Logistic_Regression_kaggle(test_data, X, y, listing_id)
    # SVM_kaggle(test_data, X, y, listing_id)
    # data.to_csv('testtesttest.csv', index=False)
    # Decision_Tree(data, X, y)
    # Logistic_Regression_score(X, y)
    SVM_score(X, y)
    # Nearest_Subway(data)
