import pandas as pd
from collections import Counter

# 5개의 CSV 파일 이름
csv_files = ['test.csv', 'test.csv', 'test3.csv', 'test4.csv', 'test5.csv']

# 동일한 ID에 대한 Hard Voting 결과를 저장할 딕셔너리 생성
voting_results = {}

# 각 CSV 파일을 읽어서 하드 보팅
for file in csv_files:
    df = pd.read_csv(file)

    for unique_id in df['id'].unique():
        # 동일한 ID에 대한 Sentiment 값 추출
        sentiment_values = df[df['id'] == unique_id]['sentiment'].values

        # Sentiment 값들에 대한 Hard Voting 수행
        sentiment_counts = Counter(sentiment_values)
        majority_sentiment = sentiment_counts.most_common(1)[0][0]

        # Hard Voting 결과를 딕셔너리에 저장
        if unique_id in voting_results:
            voting_results[unique_id].append(majority_sentiment)
        else:
            voting_results[unique_id] = [majority_sentiment]

# 결과를 DataFrame으로 변환
result_df = pd.DataFrame(columns=['id', 'sentiment'])
for unique_id, sentiments in voting_results.items():
    sentiment_counts = Counter(sentiments)
    majority_sentiment = sentiment_counts.most_common(1)[0][0]
    result_df = pd.concat([result_df, pd.DataFrame({'id': [unique_id], 'sentiment': [majority_sentiment]})],
                          ignore_index=True)

# 최종 결과를 새로운 CSV 파일로 저장
result_df.to_csv('hard_voting_result.csv', index=False)
