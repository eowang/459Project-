import pandas as pd
import sys
import matplotlib.pyplot as plt
#import cv2 

def count_null_rows(row):
	null_count = 0
	if row == '0':
		null_count += 1
	elif row == '':
		null_count += 1
	elif row == []:
		null_count += 1
	elif row == ' ':
		null_count += 1
	else:
		null_count += 0

	return null_count

def Missing_Values(data):
	null_bathrooms = data['bathrooms'].apply(lambda x:count_null_rows(x))
	result = pd.DataFrame(null_bathrooms)
	data.rename(columns={"bathrooms": "null_bathrooms"})

	result['null_bedrooms'] = data['bedrooms'].apply(lambda x:count_null_rows(x))

	result['null_building_id'] = data['building_id'].apply(lambda x:count_null_rows(x))

	result['null_created'] = data['created'].apply(lambda x:count_null_rows(x))

	result['null_description'] = data['description'].apply(lambda x:count_null_rows(x))

	result['null_interest_level'] = data['interest_level'].apply(lambda x:count_null_rows(x))

	result['null_latitude'] = data['latitude'].apply(lambda x:count_null_rows(x))

	result['null_listing_id'] = data['listing_id'].apply(lambda x:count_null_rows(x))

	result['null_longitude'] = data['longitude'].apply(lambda x:count_null_rows(x))

	result['null_manager_id'] = data['manager_id'].apply(lambda x:count_null_rows(x))

	result['null_photos'] = data['photos'].apply(lambda x:count_null_rows(x))

	result['null_price'] = data['price'].apply(lambda x:count_null_rows(x))

	result['null_street_address'] = data['street_address'].apply(lambda x:count_null_rows(x))

	result['null_display_address'] = data['display_address'].apply(lambda x:count_null_rows(x))

	result['null_features'] = data['features'].apply(lambda x:count_null_rows(x))

	total_missing = result.sum(axis=0)
	print(total_missing)

def Outliers(data):
	#box plot of numeric values to find outliers
	plt.boxplot(data['bathrooms'], notch=True, vert=False)
	plt.title('boxplot of number of bathrooms')
	plt.show()

	plt.boxplot(data['bedrooms'], notch=True, vert=False)
	plt.title('boxplot of number of bedrooms')
	plt.show()

	plt.boxplot(data['price'], notch=True, vert=False)
	plt.title('boxplot of prices')
	plt.show()

	plt.boxplot(data['latitude'], notch=True, vert=False)
	plt.title('boxplot of latitude')
	plt.show()

	plt.boxplot(data['longitude'], notch=True, vert=False)
	plt.title('boxplot of longitude')
	plt.show()

	#impute price outliers
	bedrooms_averages = data.groupby('bedrooms').agg({'price':'mean'}).to_dict()
	bedrooms_averages = bedrooms_averages['price']
	
	apt_values = data[data.price.values > 70000]
	apt_values['price'] = apt_values['bedrooms'].map(bedrooms_averages)

	data = data[data.price.values < 70000]
	data = data.append(apt_values)

	plt.boxplot(data['price'], notch=True, vert=False)
	plt.title('boxplot of prices after imputation')
	plt.show()

	#impute longitude and latitude
	# data = data[(data.longitude.values > -80) | (data.longitude.values < -85)]
	# data = data[(data.latitude.values > 40) | (data.latitude.values < 42)]

	return data

def Exploratory_Data_Analysis(data):  
	##plotting histograms
	plt.hist(data['price'], bins=50)
	plt.show()

	#plotting new york area postings
	data = data[(data.longitude.values > -74.3) & (data.longitude.values < -73.7)]
	data = data[(data.latitude.values > 40.4) & (data.latitude.values < 41)]
	plt.hist(data['longitude'], bins=40)
	plt.show()
	plt.hist(data['latitude'],bins=40)
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

	#Show proportion of target variable values 
	data['interest_level'].value_counts().plot(kind='bar')
	plt.show()

def Image_Feature_Extraction(data):
	#Gray scale image 
	img = cv2.imread('image1.jpg',0)
	cv2.imshow('image1',img)
	cv2.waitKey()

	#histogram showing intensity of image (0-255, 0 = dark, 255 = bright)
	plt.hist(img.ravel(),256,[0,256])
	plt.show()
	
	#RGB image 
	img2 = cv2.imread('image1.jpg')
	cv2.imshow('image1',img2)
	cv2.waitKey()

	#2D histogram showing Hue & Saturation
	hsv = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1], None, [180, 50], [0, 180, 0, 50])
	plt.imshow(hist,interpolation = 'nearest')
	plt.show()
	
def Text_Feature_Extraction(data):
	#extract word count
	data['word_count'] = data['description'].str.count(' ') + 1
	data.to_csv('raw_data.csv', index=False)

	#extract most popular features
	all_feature_words = sum(data.features, [])
	feature_word_counts =  Counter(all_feature_words)
	top_words = [word for word,cnt in feature_word_counts.most_common(70)]
	data['common_features'] = data['features'].apply(lambda x: [word for word in x if word in top_words])


if __name__ == '__main__':
	data = pd.read_json(sys.argv[1])
	#Missing_Values(data)
	#data = Outliers(data)
	Exploratory_Data_Analysis(data)
	#Image_Feature_Extraction(data)
	#Text_Feature_Extraction(data)
