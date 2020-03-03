import pandas as pd
import sys
import matplotlib.pyplot as plt 
import numpy as np

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