import pandas as pd
import sys
import matplotlib.pyplot as plt
from collections import Counter
import cv2
import numpy as np


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
    plt.title('box plot of number of bathrooms')
    plt.show()

    plt.boxplot(data['bedrooms'], notch=True, vert=False)
    plt.title('box plot of number of bedrooms')
    plt.show()

    plt.boxplot(data['price'], notch=True, vert=False)
    plt.title('box plot of prices before imputation')
    plt.show()

    plt.boxplot(data['latitude'], notch=True, vert=False)
    plt.title('box plot of latitude before imputation')
    plt.show()

    plt.boxplot(data['longitude'], notch=True, vert=False)
    plt.title('box plot of longitude before imputation')
    plt.show()

    #count and impute price outliers
    bedrooms_averages = data.groupby('bedrooms').agg({'price':'mean'}).to_dict()
    bedrooms_averages = bedrooms_averages['price']
    
    apt_values = data[data.price.values > 70000]
    apt_values['price'] = apt_values['bedrooms'].map(bedrooms_averages)
    print("The number of price outliers is", apt_values.shape[0])

    data = data[data.price.values < 70000]
    data = data.append(apt_values)

    #count and impute logitude and latitude outliers
    count_before_impute = data.shape[0]
    data = data[(data.longitude.values > -74.26) & (data.longitude.values < -73.69)]
    data = data[(data.latitude.values > 40.49) & (data.latitude.values < 40.93)]
    print("the number of longitude and latitude outliers is", (count_before_impute - data.shape[0]))

    #boxplots after imputation
    plt.boxplot(data['price'], notch=True, vert=False)
    plt.title('box plot of prices after imputation')
    plt.show()

    plt.boxplot(data['latitude'], notch=True, vert=False)
    plt.title('box plot of latitude after imputation')
    plt.show()

    plt.boxplot(data['longitude'], notch=True, vert=False)
    plt.title('box plot of longitude after imputation')
    plt.show()

    return data

def Exploratory_Data_Analysis(data):
    ##plotting histograms
    plt.hist(data['price'], bins=50)
    plt.title('Histogram of Prices')
    plt.xlabel('Listing Price per Month (USD)')
    plt.ylabel('Count')
    plt.show()

    #plotting longitude and latitude
    plt.hist(data['longitude'], bins=40)
    plt.title('Histogram of Longitude')
    plt.xlabel('Longitude(°)')
    plt.ylabel('Count')
    plt.show()

    plt.hist(data['latitude'],bins=40)
    plt.title('Histogram of Latitude')
    plt.xlabel('Latitude(°)')
    plt.ylabel('Count')
    plt.show()

    ##plot hour-wise listing trend and find top 5 busiest hours
    data['created'] = pd.to_datetime(data['created']) #double check that this converts AM/PM to 24hr time
    data['hour_created'] = data['created'].dt.hour
    conversions = {'low':1,'medium':2,'high':3}
    data['numeric_interest_level'] = data['interest_level'].map(conversions)
    avg_interest_by_hour = data.groupby('hour_created', as_index=False)['numeric_interest_level'].mean()
    plt.plot(avg_interest_by_hour['hour_created'], avg_interest_by_hour['numeric_interest_level'], 'b', alpha=0.5)
    plt.title('Hour-Wise Listing Trend')
    plt.xlabel('Hour')
    plt.ylabel('Average Interest Level')
    plt.show()
    top_five_hours = avg_interest_by_hour.sort_values('numeric_interest_level').head(5)
    print("the top 5 busiest hours of postings are:", top_five_hours['hour_created'].values.tolist())

    #Show proportion of target variable values
    data['interest_level'].value_counts().plot(kind='bar')
    plt.title('Proportion of Interest Levels')
    plt.show()

def Image_Feature_Extraction(data):
    #plotting histograms
    plt.hist(data['price'])
    plt.xlabel('Listing Price per Month (USD)')
    plt.ylabel('Count')
    plt.show()

    plt.hist(data['longitude'])
    plt.xlabel('Longitude(°)')
    plt.ylabel('Count')
    plt.show()

    plt.hist(data['latitude'])
    plt.xlabel('Latitude(°)')
    plt.ylabel('Count')
    plt.show()

    ##plot hour-wise listing trend and find top 5 busiest hours
    data['created'] = pd.to_datetime(data['created']) #double check that this converts AM/PM to 24hr time

    data['hour_created'] = data['created'].dt.hour
    conversions = {'low':1,'medium':2,'high':3}
    data['numeric_interest_level'] = data['interest_level'].map(conversions)
    avg_interest_by_hour = data.groupby('hour_created', as_index=False)['numeric_interest_level'].mean()
    plt.plot(avg_interest_by_hour['hour_created'], avg_interest_by_hour['numeric_interest_level'], 'b', alpha=0.5)
    plt.xlabel('Hour')
    plt.ylabel('Average interest')
    plt.show()
    top_five_hours = avg_interest_by_hour.sort_values('numeric_interest_level').head(5)
    print("the top 5 busiest hours of postings are:", top_five_hours['hour_created'].values.tolist())

    #Show proportion of target variable values
    plt.hist(data['interest_level'])
    plt.xlabel('')
    plt.ylabel('')
    plt.show()

    #Extract image features
    
    #Aggregate images together
    img2 =cv2.imread('image1.jpg')
    img3= cv2.imread('image2.jpg')
    img4= cv2.imread('image3.jpg')
    img5= cv2.imread('image4.jpg')
    img6= cv2.imread('image5.jpg')

    width, height, channels = img2.shape

    img3 = cv2.resize(img3, (height, width))
    img4 = cv2.resize(img4, (height, width))
    img5 = cv2.resize(img5, (height, width))
    img6 = cv2.resize(img6, (height, width))

    vis = np.concatenate((img2, img3,img4,img5,img6), axis=1)
    cv2.imwrite('fin.jpg',vis)

    #Grayscale histogram of aggregate photo  (0-255, 0 = dark, 255 = bright)
    grayAggr = cv2.imread('fin.jpg', 0)

    plt.hist(grayAggr.ravel(),256,[0,256])
    plt.xlabel('Intensity of aggregate image (0-255)')
    plt.ylabel('Number of pixels')
    plt.show()

    #2D histogram showing Hue & Saturation of aggregate photo
    rgbAggr = cv2.imread('fin.jpg')

    hsv1 = cv2.cvtColor(rgbAggr,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv1], [0, 1], None, [180, 50], [0, 180, 0, 50])
    plt.imshow(hist,interpolation = 'nearest')
    plt.xlabel('Hue of aggregate image (0-180)')
    plt.ylabel('Saturation of aggregate image(0-256)')
    plt.show()
    
def Text_Feature_Extraction(data):
    #extract word count
    data['word_count'] = data['description'].str.count(' ') + 1

    #extract most popular features
    all_feature_words = sum(data.features, [])
    feature_word_counts =  Counter(all_feature_words)
    top_words = [word for word,cnt in feature_word_counts.most_common(70)]
    data['common_features'] = data['features'].apply(lambda x: [word for word in x if word in top_words])
    data.to_csv('raw_data.csv', index=False)

if __name__ == '__main__':
    data = pd.read_json(sys.argv[1])
    Missing_Values(data)
    data = Outliers(data)
    Exploratory_Data_Analysis(data)
    Image_Feature_Extraction(data)
    Text_Feature_Extraction(data)
