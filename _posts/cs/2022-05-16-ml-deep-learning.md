---
layout: post
title: Deep Learning 기초
author: An Hyungjun
categories: [AI]
tags: [ai, summary, deep learning, ml]
---

Deep Learning 기초 지식을 정리한다.

# 기초 용어

## 딥러닝
- 머신러닝의 여러 방법론 중 하나로 인공신경망에 기반함
- 인공지능 > 머신러닝 > 딥러닝
- 60 ~ 80년대에 이론적 배경은 정립되었으나 H/W 발달 등에 힘입어 2000년대에 들어 활발히 사용됨
- 결국은 손실함수(Loss Function)의 최적화가 목적

## 인공 신경망
- 사람의 신경 시스템을 모방한 학습 알고리즘
- 모델 스스로 데이터의 특성을 학습함(지도, 비지도 학습 모두 가능함)
- 명시적 프로그래밍의 한계(ex. 자율 주행)를 극복할 수 있음

## 퍼셉트론
- 인공 신경망의 한 종류
- 가중치, 편향(Bias), 비선형의 활성화(activation) 함수로 이뤄짐
- 활성화 함수를 사용하는 이유
	- 노이즈 제거 또는 학습 편의

### 가중치
- 노드간의 연결강도를 의미

### 선형 분류기
- 단층 퍼셉트론(1개 Input Layer + 1개 Output Layer로 구성)을 통해 선형 분류기를 구현할 수 있음
- 예시: AND, OR, NAND, NOR

### 비선형 문제
- 다층 퍼셉트론(Hidden Layer가 추가 되어 있음)을 통해 해결
- 예시: XOR

## Backpropogation(역전파)
- 복잡한 레이어의 신경망에서 가중치의 기울기를 계산할 수 있게 하는 방법
- 오차값을 뒤로 전파하면서 변수를 갱신하는 알고리즘

# TensorFlow
- 현재(2022년) 가장 널리 쓰이는 딥러닝 프레임워크
- 유연성, 효율성, 확장성

## Tensor
- Multidimensional Arrays(Data)
- 다차원 배열로 나타내는 데이터를 의미

## Flow
- 데이터의 흐름을 의미
- TensorFlow에서 계산은 데이터 플로우 그래프로 수행됨(그래프를 따라 데이터가 노드를 거쳐감)

## TF1 vs. TF2
- TF1에서는 그래프, 세션 등 비직관적인 개념이 있었음

## TensorFlow 기초 사용

### 자료형
```python
tf.float32	# 32-bit float
tf.float64	# 64-bit float
tf.int8		# 8-bit integer
tf.int16	# 16-bit integer
tf.int32	# 32-bit integer
tf.uint8	# 8-bit unsigned integer
tf.string	# String
tf.bool		# boolean
```

### Tansor의 종류
```python
#-------------------------------------------------------------
# Constant Tensor
#-------------------------------------------------------------

# 특정 값
tensor_a = tf.constant(value, dtype=None, shape=None, name=None)

# 모든 값이 0
tensor_b = tf.zeros(shape, dtype=tf.float32, name=None)

# 모든 값이 1
tensor_c = tf.ones(shape, dtype=tf.float32, name=None)

#-------------------------------------------------------------
# Sequence Tensor
#-------------------------------------------------------------

# start에서 stop까지 증가하는 num 개의 데이터
tensor_d = tf.linspace(start, stop, num, name=None)

# start에서 limit까지 delta씩 증가
tensor_e = tf.range(start, limit, delta, name=None)

#-------------------------------------------------------------
# Variable Tensor
#-------------------------------------------------------------

tensor_f = tf.Variable(initial_value=None, dtype=None, name=None)
```

### Tansor 사칙 연산
```python
tf.add(a, b)
tf.subtract(a, b)
tf.multifly(a, b)
tf.truediv(a, b)
```

## 딥러닝 모델 구현 

### 데이터 셋 준비
- Epoch: 전체 데이터 셋에 대해 1회 학습을 완료한 상태
- Batch(mini-batch): 나눠진 데이터 셋
- iteration: 전체 데이터 셋을 몇개의 Batch로 나눴는가

### Keras
- 모델 생성을 돕는 고수준 API

```python
# 모델 클래스 개체 생성
tf.keras.models.Sequential()

# 모델의 각 Layer 구성
tf.keras.layers.Dense(units, activation)
	# units: 레이어 안의 Node의 수
	# activation: 활성화 함수 설정

# Layer 추가
m.add()

# 최적화 방식 설정
model.compile(optimizer, loss)

# 학습
model.fit(X_train, y_train)

# 평가
model.evaluate(X_test, y_test)

# 예측
model.predict(X)
```

- Layer 구성 방법 예시

```python
# Ver.1
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_dim=2, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# Ver.2
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, input_dim=2, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  
```

# 딥러닝의 문제점(Second AI Winter)
- 학습 속도: 데이터가 많아지면서 학습 시간이 기하급수적으로 늘어남
- 기울기 소실: 출력값과 멀어질 수록 기울기가 0에 가까워짐 -> 출력값에서 먼 노드의 가중치가 잘 변하지 않음
- 초기값 설정: 초기값 설정에 따라 성능차이가 큼
- 과적합: 보통 모델이 복잡해질 수록 과적합 가능성이 높아짐

## 학습 속도 문제 해법
- SGD(Stochastic Gradient Descent) 등을 이용해서 전체 데이터가 아닌 부분 데이터(Mini-batch)만 활용해서 손실 함수를 계산함
- SGD의 파생 알고리즘
	- Momentum: 과거 기울기를 나중에 일정 부분 사용함
	- AdaGrad(Adaptive Gradient): 변수들의 변화 정도에 따라 Learning rate를 차등 적용함
	- RMSProp: AdaGrad의 개선 버전으로 과거 기울기를 잊고 새 기울기 정보를 크게 반영
	- Adam(Momentum + RMSProp): 가장 발전된 최적화 알고리즘

### 다양한 최적화 함수

```python
tf.keras.optimizers.SGD(lr=0.01, momentum=0)

tf.keras.optimizers.Adagrad(lr=0.01, epsilon=0.00001, decay=0.4)
	# epsilon: 연산 시 devide by zero를 막음
	# decay: 업데이트마다 학습률을 비율만큼 줄여주는 파라미터

tf.keras.optimizers.RMSprop(lr=0.01)

tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
	# beta_1: 모멘텀
	# beta_2: step_size

```

### 손실 함수 binary crossentropy
- 두 확률 분포간의 차이를 측정하는 손실 함수
- 작을 수록 예측결과가 좋음을 의미

```python
# 손실 함수의 점수 출력 예시
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy', 'binary_crossentropy'])
model.summary()
history = model.fit(train_data, train_labels, epochs = 20, batch_size = 50, validation_data = (test_data, test_labels), verbose = 0)
scores = history.history['val_binary_crossentropy'][-1]
print('scores: ', scores)
```

## 기울기 소실 문제 해법
- Layer가 깊을 수록 sigmoid 함수 말고 다른 함수를 활성화 함수로 사용하기

### 기울기 소실 문제의 주된 발생 원인
- 주로 사용되는 Sigmoid 함수가 입력의 절대값이 클 수록 기울기가 0에 가까워 짐
- Chain Rule에 의해 미분값이 계속 곱해지면서 점점 0이 됨

### 해법
- Hidden Layer에서 활성화 함수로 ReLU를 사용
- Output Layer에서 활성화 함수로 Tanh를 사용

## 초기값 설정 문제 해법
- 2022년 기준 He + ReLU가 가장 널리 쓰임

### 발달 과정
- 표준 정규 분포를 이용: 결국 추가 학습이 안 되는 시점이 발생
- 표준 편차를 줄임
- Xavier 초기화 방법(표준 정규 분포를 입력 개수의 제곱근으로 나눔) + Sigmoid를 사용
- He 초기화 방법(표준 정규 분포를 입력 개수 절반의 제곱근으로 나눔) + ReLU

## 과적합 방지

### 아이디어
- 정규화(모델이 복잡할 수록 가중치 합이 커지는 경향이 있음): 기존 손실 함수에 규제항을 더함

### L1 정규화(Lasso Regularization)
- 가중치의 절대값의 합을 규제 항으로 정의
- 작은 가중치들이 거의 0으로 수렴하여 몇개의 중요 가중치만 남김

### L2 정규화(Ridge Regularization)
- 가중치의 제곱의 합을 규제항으로 정의
- L1에 비해 0으로 수렴하는 가중치가 적으나 큰 값을 가진 가중치를 제약하는 효과가 있음

### 드롭아웃(DropOut)
- 각 Layer마다 일정 비율의 뉴런을 임의로 Drop시킴

### 배치 정규화(Batch Normalization)
- Normalization을 처음 input data 뿐만 아니라 신경망 내부 Hidden Layer의 input에도 적용
- 매 Layer마다 정규화를 하므로 가중치 초기값에 의존하지 않음

# Keras Template
```python
model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=(28, 28)),

	# Actiation: Sigmoid
	tf.keras.layers.Dense(128, activation='sigmoid'),

	# Actiation: ReLU
	tf.keras.layers.Dense(128, activation='relu'),

	# Actiation: tanh
	tf.keras.layers.Dense(128, activation='tanh'),

	# L1
	tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1(0.001)),

	# L2
	tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
	
	# Dropout
	tf.keras.layers.Dropout(0.1),

	# Batch Normalization
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Activation('relu'),
	tf.keras.layers.Dense(512),

	tf.keras.layers.Dense(10, activation='softmax')
])
```





