import pandas as pd
import sys

def preprocessing():
	data = pd.read_json(sys.argv[1])
	print("hello")
	print(data)
	data.to_csv('raw_data.csv', index=False)

if __name__ == '__main__':
	preprocessing()