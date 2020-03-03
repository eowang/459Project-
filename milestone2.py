import pandas as pd
import sys
import matplotlib.pyplot as plt 
import numpy as np

def Min_Euclidean_Dist(data1, data2, cols=['latitude','longitude']):
    data2['distance']= data2.apply(lambda x: np.square())
    return np.linalg.norm(data1[cols].values - data2[cols].values,
                   axis=1)
    

def Nearest_Subway(data):
    subwayData= pd.read_csv("subwayData.csv")
    # subwayData['coordinates'] = subwayData['the_geom'].str.split(' (').str[0]


    subwayData['longitude']= subwayData['the_geom'].apply(lambda st: st[st.find("(")+1:st.find("4")])
    subwayData['latitude']= subwayData['the_geom'].apply(lambda st: st[st.find(" 4")+1:st.find(")")])
    
    distanceArray= Min_Euclidean_Dist(data,subwayData)
    data.apply(lambda x: Min_Euclidean_Dist())


# def Feature_Selection(data):

# def Decision_Tree(data):

# def Logistic_Regression(data):

# def SVM(data):

if __name__ == '__main__':
    data = pd.read_csv('raw_data.csv')
	# data = pd.read_json(sys.argv[1])
	# Feature_Selection(data)
	# Decision_Tree(data)
	# Logistic_Regression(data)
	# SVM(data)
    Nearest_Subway(data)