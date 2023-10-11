import pandas as pd
from collections import Counter

# 5개의 CSV 파일 이름
csv_files = ['file1.csv', 'file2.csv', 'file3.csv', 'file4.csv', 'file5.csv']

# 데이터를 저장할 DataFrame 생성
result_df = pd.DataFrame(columns=['id', 'sentiment'])

# 각 CSV 파일을 읽어서 하드 보팅
for file in csv_files:
    df = pd.read_csv(file)

    for unique_id in df['id'].unique():
        # 동일한 ID에 대한 Sentiment 값 추출
        sentiment_values = df[df['id'] == unique_id]['sentiment'].values

        # Sentiment 값들에 대한 Hard Voting 수행
        sentiment_counts = Counter(sentiment_values)
        majority_sentiment = sentiment_counts.most_common(1)[0][0]

        # 하드 보팅 결과를 결과 DataFrame에 추가
        result_df = result_df.append({'id': unique_id, 'sentiment': majority_sentiment}, ignore_index=True)

# 최종 결과를 새로운 CSV 파일로 저장
result_df.to_csv('hard_voting_result.csv', index=False)
