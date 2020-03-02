import pandas as pd
import sys
import matplotlib.pyplot as plt 

def nearestSubway(data):
    subwayData= pd.read_csv("subwayData.csv")
    # subwayData['coordinates'] = subwayData['the_geom'].str.split(' (').str[0]
    subwayData['longitude']= subwayData['the_geom'].apply(lambda st: st[st.find("(")+1:st.find("4")])
    subwayData['latitude']= subwayData['the_geom'].apply(lambda st: st[st.find(" 4")+1:st.find(")")])

    subwayData.to_csv('subwayData1.csv', index=False)

def Feature_Selection(data):

def Decision_Tree(data):

def Logistic_Regression(data):

def SVM(data):

if __name__ == '__main__':
	data = pd.read_json(sys.argv[1])
	Feature_Selection(data)
	Decision_Tree(data)
	Logistic_Regression(data)
	SVM(data)