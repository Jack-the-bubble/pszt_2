import pandas as pd
import numpy as np
import matplotlib as mpt
import math
from sklearn.model_selection import cross_val_score

def analyze_data():
	data = pd.read_csv('../data/forestfires.csv')
	days = {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6}
	month = {'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5, 'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9,
			 'nov': 10, 'dec': 11}
	data['day']=data['day'].map(days)
	data['month']=data['month'].map(month)
	# print(data)
	return data


if __name__ == '__main__':
	analyze_data()