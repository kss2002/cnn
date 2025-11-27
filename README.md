과제1. CNN을 이용한 데이터 분류 모델 구현

과제 목표 : 합성곱 신경망을 이용하여 데이터세트를 분류하고, 학습 및 평가 과정을 통해 인공지능 모델의 원리를 이해한다.

과제 설명 : 

데이터 세트 : Flowers 데이터 세트 이용

1) CNN 모델 : Conv(3x3필터, 32개)-Maxpooling - Conv(3x3필터, 64개)-Maxpooling - Conv(3x3필터, 128개)-Maxpooling – Flatten() – Dense(128) – Dense(5)

2) 사전 학습 모델 : ResNet50