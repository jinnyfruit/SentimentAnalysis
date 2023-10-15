import pandas as pd
from transformers import AutoTokenizer

# CSV 파일을 읽어옴 (파일 경로에 맞게 수정)
df = pd.read_csv('train_fold_0.csv')

# 언어 모델 및 토크나이저 선택 (예: BERT)
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 'text' 열에서 토크나이즈하고 OOV 단어를 찾음
oov_words = []
for text in df['text']:
    tokens = tokenizer.tokenize(text)
    oov_words.extend([word for word in tokens if word not in tokenizer.get_vocab()])

# 중복 OOV 단어 제거 (선택 사항)
oov_words = list(set(oov_words))

# OOV 단어 출력
print("OOV 단어:", oov_words)
