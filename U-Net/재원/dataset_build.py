# 참고 : https://dacon.io/codeshare/4245
# 논문에서 언급되었던 ISBI 2012 EM Segmentation membrane dataset

"""
train-volumne.tif 파일 -> 훈련 이미지(512*512 grayscale 세포 Image) 30장
train-labels.tif 파일 -> 정답에 해당하는 Segmentation map(배경과 세포를 구분하며, 배경은 1이고 세포는 255)
test-volumne.tif 파일 -> 테스트 이미지
"""

# 패키지 불러오기
import os
import numpy as np
from PIL import Image # 이미지를 읽고 수정하고 저장하는 역할
import matplotlib.pyplot as plt

# 데이터 불러오기
dir_data = './dataset' # 상위 폴더가 dataset으로 설정되어 있어 경로 설정

name_label = 'train-labels.tif' #label 파일이름
name_input = 'train-volume.tif' #train 이미지 zip파일

img_label = Image.open(os.path.join(dir_data, name_label)) #결국 label도 이미지 형태이므로 Image 패키지로 경로 내 zip파일 읽어오기
img_input = Image.open(os.path.join(dir_data, name_input)) #train image 파일 읽어오기

ny, nx = img_label.size #기존 ny, nx였는데 PIL 패키지 이미지 width, height순서라 x를 가로로 조정하기 위해 순서 변경 
nframe = img_label.n_frames #이미지 개수 확인

# training 이미지 나눠서 넣을 train,test,val 폴더 생성
# 이미지(frame)개수 설정
nframe_train = 24 
nframe_val = 3
nframe_test = 3

# 경로명 설정(기존 dataset폴더에 하위 폴더로 train, val, test)
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

# 만약 기존에 따로 설정되지 않았다면 폴더 생성
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

# 전체 이미지 프레임 번호, 인덱스 배열을 만들고 [0,1,2...30] -> 랜덤으로 순서를 shuffle
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

# tif의 멀티 프레임 이미지를 이미지 개별 파일 하나하나로 분할해서 저장
# 개별 train 이미지를 npy 파일로 저장
# 이때 왜 npy(numpy)형태로 저장하냐? grayscale 이미지 형태이기 때문에 채널 정보가 없어서
# 초기에 train은 안띄워도 되기 때문에 offset 0으로 초기 설정해주기
offset_nframe = 0

# train 이미지 저장 지정한 개수만큼 반복
for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe]) # 섞인 랜덤 배열에서 i번째 프레임 이미지 선택(seek는 포인터만 해당 파일에 놓는 역할)
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label) #해당 인덱스 번호 label numpy array 형태로 변환
    input_ = np.asarray(img_input) #해당 인덱스 번호 train image numpy array 형태로 변환

    # 지정경로에 npy 형태로 각각 이미지, 라벨 저장
    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_) 
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

# valid도 동일하게 진행(offset을 nframe_train으로 설정하면, train 개수 그 이후부터)
offset_nframe = nframe_train

for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

## offset 27로 설정해서 나머지 3개를 test로 추출
offset_nframe = nframe_train + nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)