import pandas as pd

# 파일 경로 설정
input_csv_file = 'train0.csv'
output_csv_file = 'preprocessed_train0.csv'

# CSV 파일 불러오기
data = pd.read_csv(input_csv_file)

# 전처리 함수 정의
def preprocess(text):
    preprocessed_text = []
    for t in text.split():
        if len(t) > 1:
            t = '@user' if t[0] == '@' and t.count('@') == 1 else t
            t = 'http' if t.startswith('http') else t
        preprocessed_text.append(t)
    return ' '.join(preprocessed_text)

# 'text' 열에 전처리 적용
data['text'] = data['text'].apply(preprocess)

# 수정된 데이터를 새로운 CSV 파일에 저장
data.to_csv(output_csv_file, index=False)

print(f'전처리된 데이터를 {output_csv_file} 파일로 저장했습니다.')
