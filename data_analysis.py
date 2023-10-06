import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/train.csv')
print(data.head())

# 각 열에서 NULL 값의 개수를 확인
null_counts = data.isnull().sum()
print("Columns with NULL values:")
print(null_counts)

# 'sentiment' 열 값별 데이터 개수를 계산합니다.
sentiment_counts = data['sentiment'].value_counts()

# 각 sentiment 값의 백분율을 계산합니다.
percentage = sentiment_counts / len(data) * 100

# 결과를 출력합니다.
print("Sentiment Percentage:")
print(percentage.round(2))
