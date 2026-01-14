#!/usr/bin/env python
# coding: utf-8

# ## 리뷰 키워드 감정 분석 분류 시스템 ↓

# In[1]:


pip install konlpy tensorflow scikit-learn


# In[2]:


import pandas as pd

# CSV 파일 로드
df = pd.read_csv("review_label.csv")

# 리뷰 텍스트 리스트
texts = df["review"].tolist()

# 라벨 리스트
labels = df["label"].tolist()

# 확인
print(len(texts), len(labels))
print(texts[:3])
print(labels[:3])

# cleanup
del df


# In[3]:


# 불용어(stopwords) 정의 (중요)
stopwords = set([
    # 조사
    '이', '가', '을', '를', '의', '에', '에서', '로', '으로', '와', '과', 
    '도', '만', '까지', '부터', '한테', '에게', '께', '더러', '라고',

    # 어미
    '은', '는', '이다', '입니다', '습니다', 'ㅂ니다', '합니다', 
    '해요', '이에요', '예요', '네요', '군요', '구나', '구먼',

    # 대명사
    '저', '제', '나', '내', '우리', '저희', '너', '당신',
    '이것', '그것', '저것', '여기', '거기', '저기',

    # 관형사/부사 (의미 없는 것만)
    '그', '이', '저', '어떤', '무슨', '모든', '어느',
    '좀', '잘', '더', '덜', '매우', '아주', '조금', '많이',

    # 접속사
    '그리고', '그러나', '하지만', '또', '및', '또는', '혹은',

    # 기타 불필요한 단어
    '것', '수', '등', '및', '때', '년', '월', '일',
    '하다', '되다', '있다', '없다', '이다', '아니다',

    # 탁송 리뷰에서 의미 없는 단어들
    '이용', '서비스', '업체', '회사'  # 너무 일반적이어서 구별력 없음
])


# In[4]:


# 한글 형태소 분석 (Okt), 불용어 제거
from konlpy.tag import Okt

okt = Okt()

def tokenize(text):
    tokens = okt.morphs(text, stem=True)
    tokens = [t for t in tokens if t not in stopwords]
    return tokens

tokenized_texts = [tokenize(t) for t in texts]

print(tokenized_texts)


# In[5]:


# 단어장 생성
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
# tokenizer에 토큰을 전달하여 단어장을 생성
tokenizer.fit_on_texts(tokenized_texts)
# tokenizer에 생성된 단어장 확인
word_index = tokenizer.word_index
print(word_index)


# In[6]:


# 정수 인코딩
sequences = tokenizer.texts_to_sequences(tokenized_texts)
print(sequences)


# In[7]:


# Padding
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 10  # 한 문장에 포함된 토큰의 수가 10을 초과하는 경우가 거의 없다

X = pad_sequences(sequences, maxlen=MAX_LEN)
y = labels


# In[8]:


# 학습 / 테스트 분리
import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

X_train = np.array(X_train).astype('int32')
X_test = np.array(X_test).astype('int32')

# y_train, y_test도 동일하게 변환 (분류 모델이므로 보통 int32)
y_train = np.array(y_train).astype('int32')
y_test = np.array(y_test).astype('int32')


# In[9]:


# 모델 구성 (Embedding + LSTM)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input

#정수로 인코딩된 한개의 문장이 모델에 입력될 때는 패딩으로 추가된 0도 포함되어 있음
#그러나 위에서 생성된 단어장에는 0과 매핑된 단어는 보이지 않으므로 모델이 0의 의미는 해석 불가
#위에서 생성된 단어장에는 1부터 숫자가 배정되어 있지만 내부에서 실제 사용되는 단어장에는 0부터 시작됨
#내부에서 사용되는 실제 단어장에는 <PAD>:0 아이템도 추가되어 실제 토큰 수 + 1개가 됨
VOCAB_SIZE = len(word_index) + 1

model = Sequential([
    Input(shape=(MAX_LEN,)),      # 입력 데이터의 형태를 명시
    Embedding(VOCAB_SIZE, 64),    # 64차원벡터, 처음에는 무작위로 벡터의 원소 설정(크기,방향)
    LSTM(64),                     # 문장의 순서도 학습
    Dense(1, activation="sigmoid")# 이진분류(긍정/부정) 
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# In[10]:


# 콜백(EarlyStop, ModelCheckpoint)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. EarlyStopping: 검증 손실(val_loss)이 3회 연속 개선되지 않으면 학습을 조기 종료합니다.
early_stopping = EarlyStopping(
    monitor='val_loss',      # 감시 지표: 검증 손실
    patience=10,              # 개선되지 않아도 지켜볼 에포크 횟수
    restore_best_weights=True # 종료 후 가장 성적이 좋았던 가중치로 복구
)

# 2. ModelCheckpoint: 검증 성적이 가장 좋은 모델을 파일로 저장합니다.
checkpoint_path = "best_consignment_model.keras" # 2026년 기준 .keras 확장자 권장
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',      # 감시 지표
    save_best_only=True,     # 가장 좋은 모델만 저장
    verbose=1                # 저장 시 로그 출력
)


# In[11]:


# 학습
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint]
)


# In[12]:


history.history.keys()
acc = history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
import matplotlib.pyplot as plt
plt.figure(figsize=(5,3))
plt.plot(acc, label='accuracy')
plt.plot(loss, label='Training Loss')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('acc/loss')
plt.legend()
plt.grid(True)
plt.show()


# In[18]:


# 모델 활용
def predict_sentiment(text):
    tokens = tokenize(text)
    seq = tokenizer.texts_to_sequences([tokens])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prob = model.predict(padded)[0][0]
    return prob, "긍정" if prob >= 0.5 else "부정"

prob, label = predict_sentiment("완전 기사님과 연락이 잘 안 되어서 불안했어요 최악 ㅜㅜ")   # 부정 
print(prob, label)

prob, label = predict_sentiment("탁송 기사님이 친절하고 약속된 시간 내에 도착했어요")  # 긍정
print(prob, label)

prob, label = predict_sentiment("기사님이 인수인계를 하는 태도가 성실하지 않았어요")  # 부정
print(prob, label)

prob, label = predict_sentiment("서비스는 전반적으로 만족하지만 운행거리 미터수가 왜 그렇게 많은지 자세히 알려주면 더 좋겠어요")  # 긍정
print(prob, label)

prob, label = predict_sentiment("이 서비스는 절대로 권장하지 않습니다")  # 부정
print(prob, label)


# In[ ]:




