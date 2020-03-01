import pandas as pd
import sys
import matplotlib.pyplot as plt 

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