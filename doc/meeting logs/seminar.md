
### Seminar Day : Fully Convolutional Networks for Semantic Segmentation 
### 3rd, July

### Feedback
* Q: (김진우)CNN은 receptive field 안의 정보로 판단하는데, 만약 사람 얼굴에서 코나 눈 등이 우리가 아는 모습과는 다른 위치에 있으면 얼굴이 아니라고 인식할 수 있습니까?
* A: (전진우)receptive field 크기에 따라 다를 것 같습니다. 정확히 알아보지는 못했습니다. FCN을 이용하면 공간적인 정보를 유지하기 때문에 다른위치에
피처들이 존재한다면 얼굴이 아니라고 인식할 수 있을 것 같습니다.
* A: (김효훈 조교님)CNN의 가장 큰 단점이 바로 그런 부분이다. feature를 찾아내고 그에따라 분류하기 때문에 공간적인 특성이 사라진다. 이를 해결하기 위한 
네트워크가 최근에 발표되었다. 딥러닝의 아버지인 힌튼 교수의 논문인 capsule network 에 대해서 다음 주 발표를 하자.

***

### Seminar Day : Capsule Network and Dynamic Routing
### 9th, July

### Content
#### Batch Normalization (ICML 2015)
//yeonjee

* 딥러닝을 더 빠르게 진행하고 싶다!
* internal covariate shift
* 이 문제를 해결하는 제일 직관적인 방법은 Whitening 이라고 불리는, input data의 평균을 0, 분산이 1인 정규분포로 normalize 시키는 방법이 있다. 하지만 이 방법은 다변수의 경우 계산이 복잡해지며 training 마다 전체 데이터의 평균/분산을 구하는 것은 많은 계산량을 필요로 한다. 따라서 더욱 간단한 방법이 필요한데, 그게 바로 batch normalization이다.

* 이 BN Layer는 input data를 normalize 한다. 이를 통해 두가지를 얻을 수 있다.
* 1. learning rate를 올릴 수 있다.(converge가 빨라진다.)(5배, decay는 6배)
* 2. drop out 을 쓰지 않아도 된다. L2 weight regularization을 줄일 수 있다. (regularization 효과가 있으므로)

***


### Seminar Day : Decaying LR vs Batch Normalization
### 16th, July
### Content
#### BM3D
//yeonjee

* 많은 디노이징 네트워크 중에 가장 성능이 좋음
* nonlocal means - 한 이미지 안에서 비슷한 패턴이 나오는 부분을 고른다음 평균을 냄
* 평균을 바탕으로 디노이징을 함

* grouping - n차원의 조각들을 n+1 차원의 데이터구조로 만드는 것 -> 2D 이미지를 비슷한것 끼리 모아서 3차원으로 만듬

* k-means 클러스터링 등 방법이 있지만 matching 이라는 방법을 씀 - 교집합이 없는 부분집합으로 나누는방법,
* 매칭 - 대표를 골라서 비슷한것을 묶어주는 방법(거리는 L2 distance을 씀)
* collaborative filtering - 비슷한 블록을 찾아서 elementwise averaging 방식으로 디노이징

* PSNR 측정법 - 높을수록 깨끗한 이미지와 비슷하다는 의미
* 노이즈의 분산이 40이 넘어가면 성능이 급격하게 감소함



#### CNN
//Gisu-choi

* Conv Layer - Pooling Layer - FC Layer
* lr은 주로 0.1 또는 0.01 사용
* dropout - 오버피팅 방지, 특정 weight만 커지도록 하지 않게 하기 위해 특정 노드들을 꺼가면서 학습
* softmax - 점수를 확률로 바꿔줌. one-hot encoding 과 같이 사용


#### MRF 구현
//Jinwoo Kim

* Bay Net, MRF Net
* simulated annealing - 담금질, 최적점을 찾고자 함, 계산 오래걸림, 건너뛰는 폭을 점점 줄여가면서 최적점을 찾게됨


***
### Seminar Day : Saliency-Guided Unsupervised Feature Learning for Scene Classification
### 23th, July
### Content
#### Artificial Neural Network
//Gisu Choi


폰노이만 구조 - 순차적인 프로세스에는 적합

* 컴퓨터 비전 등은 순차적인 처리가 어려우므로 사람의 뇌의 작동을 본따서 만듬, 사람의 뉴런을 본따서 만듬
* 퍼셉트론 : cost function은 진리값 * (-1) * 노드값의 합을 사용
* Gradient Descent 방식으로 optimize

* linear function : f(x) + f(y) = f(x+y), af(x) = f(ax) 성립하는 것


#### 모바일 tensorflow
//Jinwoo Kim

1. JNI
2. TensorFlowInferenceInterface - pb파일만 있으면 텐서플로우 사용 가능
3. Tensorflow Lite
pb파일만 만들면 안드로이드에 올려서 포워딩 해서 결과 확인 가능


#### BM3D
//Yeonjee

* 2step
* 비슷한 피처들의 대표적인 블럭을 찾고 그것들을 모아서 한차원 늘린 grouping 해줌
* 다른 도메인으로 변환 -> 노이즈 제거 ->도메인 복구

* 도메인변환
* Discrete Fourier Transform : 섞인 신호들을 베이스들로 나누고자 하는 것, 균등하게 저장
* Wavelet Transform : 높은 주파수와 낮은 주파수로 나눔, 낮은 주파수일 수록 공간적인 정보보단 어느 범위의 주파수 안에 있는지 자세히 저장, 높은 주파수는 공간적인 정보 중요시

* 노이즈 제거 : hard-thresholding, wiener filter
* 작은 주파수들을 0으로 만들어버려서 디노이징
* G = 노이즈를 넣은 결과
* F = WG
* W를 구하는게 목표


***
### Seminar Day : Object Recognition from Local Scale-Invariant Features + Kullback Leibler Divergence
### 30th, July
### Content
#### Feedback
* 딥러닝 이전의 영상처리 기법들에 대해서 공부해야함
* 현재 많이 쓰이는 기법 중 하나로, 계산량이 많지만 그만큼 성능이 매우 뛰어남
* KL - divergence에 대해 다시 공부할 것


***
### Seminar Day : ImageNet Classification with Deep Convolutional Neural Networks
### 6th, August
### Content
#### TNRD - Trainable Nonlinear Reaction Diffusion

* diffusion? : image smoothing에 사용, 빛이 주변 픽셀로 확산된다고 모델링

* linear diffusion - 모든 가중치를 모든 방향으로 동일하게 적용, 모든 방향으로 동일하게 퍼져나감
* nonlinear - 모든 방향으로 동일하게 퍼져나가지 않음

* linear - 가우시안 필터링
* TV 필터링 - nonlinear, 효과좋음, 자연물은 다 nonlinear

* reaction diffusion - 두 화학물질이 서로 반응하고 섞이는 수학적 모델
* diffusion term - 화학물질이 퍼져나가는 모습을 모델링
* reaction term - 모든 화학물질이 어느 방향으로 이동하면서 반응하려고 하는지(벡터같은 역할)

* 평평한곳에선 확산이 빨리, 경계에선 느리게확산. 
* 문제점 - 언제 멈춰야할지 모름, psnr이 높아도 사람눈엔 좋지 않은 이미지일 수 있음


다음주 optimization 기법들, dropout 에 대해발표

***
