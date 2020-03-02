import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

#apply SelectKBest class to extract top 10 best features
def chiSquared(X,y):
    bestfeatures = SelectKBest(score_func=chi2, k=8)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(10,'Score'))  #print 10 best features

def featureImportance(X,y):
    model = ExtraTreesClassifier()
    model.fit(X,y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()

def heatMap(data):
    #get correlations of each features in dataset
    corrmat = data.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    #plot heat map
    g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")      
    plt.show()

#Recursive feature elimination
def recursiveFeature(X,y):
    # feature extraction
    X_scaled = preprocessing.scale(X)
    model = LogisticRegression(solver='lbfgs')
    rfe = RFE(model, 3)
    fit = rfe.fit(X_scaled, y)
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)

def numericalFeatureSelection(data):
    # data = data[['bathrooms','bedrooms','latitude','listing_id','longitude','price','hour_created','word_count','numeric_interest_level']]
    data = data.drop(data.columns[[2, 3,4,5,6,10,11,13]], axis=1)
    cols = list(data)
    cols.insert(-1, cols.pop(cols.index('interest_level')))
    cols.insert(-1, cols.pop(cols.index('Subway')))
    data = data[cols]
    data['longitude']= data['longitude'].abs()
    

    X = data.iloc[:,0:79]  #independent columns
    y = data.iloc[:,-1]    #target column i.e price range
    data.to_csv('test_data.csv', index=False)
    chiSquared(X,y) 
    featureImportance(X,y)
    heatMap(data)
    recursiveFeature(X,y)


if __name__ == '__main__':
    data = pd.read_csv("raw_data.csv")
    numericalFeatureSelection(data)


