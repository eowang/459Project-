import pandas as pd
import sys
import matplotlib.pyplot as plt

def Missing_Values_Outliers(data):
	#data.to_csv('raw_data.csv', index=False)
	return data
	
def Exploratory_Data_Analysis(data):  
	data = data.iloc[:20] #remove when working with full dataset
	##plotting histograms
	plt.hist(data['price'])
	plt.show()
	plt.hist(data['longitude'])
	plt.show()
	plt.hist(data['latitude'])
	plt.show()

	##plot hour-wise listing trend and find top 5 busiest hours
	data['created'] = pd.to_datetime(data['created']) #double check that this converts AM/PM to 24hr time
	data['hour_created'] = data['created'].dt.hour
	conversions = {'low':1,'medium':2,'high':3}
	data['numeric_interest_level'] = data['interest_level'].map(conversions)
	avg_interest_by_hour = data.groupby('hour_created', as_index=False)['numeric_interest_level'].mean()
	plt.plot(avg_interest_by_hour['hour_created'], avg_interest_by_hour['numeric_interest_level'], 'b', alpha=0.5)
	plt.show()
	top_five_hours = avg_interest_by_hour.sort_values('numeric_interest_level').head(5)
	print("the top 5 busiest hours of postings are:", top_five_hours['hour_created'].values.tolist())

if __name__ == '__main__':
	data = pd.read_json(sys.argv[1])
	data = Missing_Values_Outliers(data)
	Exploratory_Data_Analysis(data)
