import pandas as pd
import sys
import matplotlib.pyplot as plt 
import numpy as np
import statistics
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

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

# def Decision_Tree(data):

def Logistic_Regression(data):
    data = data.head(300)

    X = data.drop(['numeric_interest_level','interest_level', 'photos', 'description', 'features', 'listing_id', 'display_address', 'building_id', 'created', 'street_address', 'manager_id'], axis=1)
    y = data['numeric_interest_level']
    kf = KFold(n_splits = 5)
    scores = []

    KFold(n_splits=2, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X):
        #print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

        logistic_model = LogisticRegression()
        logistic_model.fit(X_train, y_train)

        predicted_prob = logistic_model.predict_proba(X_valid)
        scores.append(log_loss(y_valid, predicted_prob))

    avg_score = statistics.mean(scores)
    print("Scores are:", scores)
    print("Average score is:", avg_score)

def SVM(data):
    #remember to remove this later
    data = data.head(300)

    X = data.drop(['interest_level', 'photos', 'description', 'features', 'listing_id', 'display_address', 'building_id', 'created', 'street_address', 'manager_id'], axis=1)
    y = data['interest_level']
    kf = KFold(n_splits = 5)
    scores = []

    KFold(n_splits=2, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X):
        #print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

        svm_model = SVC(C=10, kernel='linear', probability=True)
        svm_model.fit(X_train, y_train)

        predicted_prob = svm_model.predict_proba(X_valid)
        scores.append(log_loss(y_valid, predicted_prob))

    avg_score = statistics.mean(scores)
    print("Scores are:", scores)
    print("Average score is:", avg_score)

if __name__ == '__main__':
    data = pd.read_csv('data_after_M1.csv')
    # data = pd.read_json(sys.argv[1])
    # Feature_Selection(data)
    # Decision_Tree(data)
    # Logistic_Regression(data)
    SVM(data)
    # Nearest_Subway(data)