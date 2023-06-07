## DACON : [[computer vision] 이상치 탐지 알고리즘 경진대회](https://dacon.io/competitions/official/235894/overview/description)

<img alt="Html" src ="https://img.shields.io/badge/dacon Final rank-Top 8%25-lightblue?style=for-the-badge"/>

#### 사물의 종류와 상태를 분류하는 컴퓨터 비전 알고리즘 개발 (2022.04.01 - 2022.05.27) - 길다영, 김채현
#### 📊 [Private] Score 0.85027 : 상위 10% (36/481)
#### 📊 [Public] Score 0.8466 : 상위 10% (46/481)

<br><br>

### Data Augmentation 방법

- SMOTETomek 기법 사용하여 기존 train_df.csv(라벨링) 파일의 라벨들의 수를 늘린다.

<p align="center">
<img src="https://user-images.githubusercontent.com/53934639/166857289-60777525-4d96-4cf6-9e06-07b59b90594e.png" width="800px"></p>

<br>

- 라벨별로 Oversampling 또는 Undersampling 해야 하는 개수를 세고, 그에 맞춰 train img들을 늘리거나 줄이면 된다.
  ![image](https://user-images.githubusercontent.com/53934639/166857431-2b8d4998-e10d-43f0-9fea-59b0330896dc.png)
  - 기존 train_img들을 복사해서 라벨별로 dir을 나눠 각 이미지들을 집어넣는다.
  - **Oversampling의 경우**, Oversampling 해야 하는 개수가 N이라면
    - 단순 image augmentation을 통해 N개의 이미지를 생성하는 것이 아니라, 먼저 기존의 라벨들이 무작위로 k개 삭제되는 원리이기 때문에
    - 라벨에 맞춰 train data에서 **k개의 이미지를 삭제**한 후, **(삭제된 개수(k) + Oversampling 해야 하는 개수 (N))개의 이미지를 새로 생성**해야 한다.
  - **Undersampling의 경우**, Undersampling 해야 하는 개수 N이라면 라벨에 맞춰 N개의 기존 train 이미지들을 삭제하면 된다.
  
  - **이미지 삭제해야 하는 경우,**
    - 기존 train data에서 라벨에 맞춰 이미지를 삭제한다. 즉, 라벨과 비교했을 때 라벨엔 없는데 train data에는 있는 이미지들을 삭제한다.
  - **이미지 생성해야 하는 경우,**
    - 라벨에 맞춰 img augmentation을 진행하여 이미지를 생성한다. 즉, 라벨과 비교했을 때 라벨에는 있는데 train data에는 없는 이미지들을 새로 만드는 것이다. 이미지들을 만들 때는 단순 복사가 아니라, 아래 train img augmentation 방법을 통해 생성한다.
    - 앞서 만든 라벨별 dir을 사용한다. 새로 생성할 이미지의 라벨 dir에서 이미지를 가져와 augmentation 시킨 후, 결과물을 train data에 집어넣는 것이다.

<br>

### Train Img Augmentation 방법
  - 이미지 기울이기, center crop + 이미지 기울이기, center crop + 왼쪽으로 늘리며 기울이기, center crop + 오른쪽으로 늘리며 기울이기, 왼쪽으로 늘리며 기울이기, 오른쪽으로 늘리며 기울이기, 이미지 기울이기 + blur 추가, bilateralFilter 추가하기 + 이미지 기울이기
  - (여기서부터 metal_nut 제외) 좌우반전 + 이미지 기울이기, 상하반전 + 이미지 기울이기, 좌우반전, 상하반전, 좌우반전 + 이미지 기울이기 + centercrop, 상하반전 + 이미지 기울이기 + centercrop, 이미지 기울이기 + blur 추가+좌우반전, 이미지 기울이기 + blur 추가 + 상하반전
    - metal_nut img의 경우 좌우상하 반전이 들어가면 정상 데이터와 비정상 데이터가 섞일 우려가 있어 제외시킴.


<br>


### Image Augmentation 결과
  <p align="center">
<img src="https://user-images.githubusercontent.com/53934639/166858518-bd0a2bde-7aee-44ce-8c10-547cc1ba67f5.png" width="800px"></p>

  

<br>

### baseline-efficientnet_b4.ipynb
- data_augmentation.ipynb에서 만든 라벨들과 train_img들을 가져와 훈련시키는 코드이다.
- SMOTETomek을 통해 새로 만든 라벨들이 train_img의 순서와 일치하지 않기 때문에 다음 코드를 통해 순서를 일치시킨다.

```y = pd.read_csv('smotetomek_result.csv')
print(y)
y['0'] = y['0'].astype(str)
y['0'] +='.png'
y['0'] = y['0'].str.replace('.0.png', '.png')

y = y.sort_values(by=['0'])
print("---------------------------------")
print(y)
train_labels = y['1']
```
<p align="center">
<img src="https://user-images.githubusercontent.com/53934639/166859767-117a05c5-0152-4cb6-90ef-13046dc407d9.png" width="300px"></p>


<br>


### 기존 baseline으로 실험 결과, 성능 비교
  - MobileNetv2 < efficientNet_b0 < MNASNet < MobileNetv3 < EfficientNet_b4
  - optimizer = NAdam

- 따라서, **efficientnet_b4, optimizer = NAdam** 사용

- **batch size = 12**로 조정
  - Batch size : training data를 여러 작은 그룹으로 나누었을 때 한 그룹에 속하는 데이터 수를 의미
  - 이 코드에선 SMOTETomek에 의해 실행할 때마다, train data 개수가 달라지지만 마지막 실행시 train img 개수 = 12588 였다.
  - 12588 = 12 * 1049 이기 때문에 딱 떨어지는 12로 설정했더니 batchsize = 32였을 때보다 정확도가 상승했다
    - 0.8152253422 -> 0.8309890947
