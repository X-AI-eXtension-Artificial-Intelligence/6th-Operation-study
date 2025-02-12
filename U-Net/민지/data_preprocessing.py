import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import kagglehub
from PIL import Image

def download_kaggle_dset(path):
    # Kaggle dataset 다운로드
    os.system(f"kaggle datasets download -d santhoshkumarv/dog-segmentation-dataset -p {path} --unzip")
    print("Dataset downloaded to:", path)


def center_crop(image_array, size):
    h, w = image_array.shape[:2]  # 이미지의 높이와 너비 구하기

    # 크롭할 영역의 시작 인덱스 계산
    start_x = (w - size) // 2
    start_y = (h - size) // 2

    # 이미지를 정중앙에서 크롭
    cropped_image = image_array[start_y:start_y + size, start_x:start_x + size]

    return cropped_image
def data_preprocessing(data_dir):
    image_lst = glob.glob(os.path.join(data_dir, 'Dog Segmentation', 'Images', '*'))

    label_lst = glob.glob(os.path.join(data_dir, 'Dog Segmentation', 'Labels', '*'))

    ## train/test/val 폴더 생성
    nframe = len(image_lst)

    nframe_train = round(nframe * 0.7)
    nframe_val = round(nframe * 0.1)
    nframe_test = round(nframe * 0.2)

    dir_save_train = os.path.join(data_dir, 'train')
    dir_save_val = os.path.join(data_dir, 'val')
    dir_save_test = os.path.join(data_dir, 'test')

    os.makedirs(dir_save_train, exist_ok=True)
    os.makedirs(dir_save_val, exist_ok=True)
    os.makedirs(dir_save_test, exist_ok=True)

    images = []
    labels = []

    for image, label in zip(image_lst, label_lst):
        img_label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
        img_input = cv2.imread(image)

        # 크롭: 중심에서 정사각형으로 자르기
        smaller_side = min(img_label.shape[0], img_label.shape[1], img_input.shape[0], img_input.shape[1])
        img_label_cropped = center_crop(img_label, smaller_side)
        img_input_cropped = center_crop(img_input, smaller_side)

        label_ = np.asarray(img_label_cropped)
        input_ = np.asarray(img_input_cropped)

        images.append(input_)
        labels.append(label_)

    # 데이터 분할
    train_images = images[:nframe_train]
    val_images = images[nframe_train:nframe_train + nframe_val]
    test_images = images[nframe_train + nframe_val:]

    train_labels = labels[:nframe_train]
    val_labels = labels[nframe_train:nframe_train + nframe_val]
    test_labels = labels[nframe_train + nframe_val:]

    # train 데이터를 저장
    for i, (img_input, img_label) in enumerate(zip(train_images, train_labels)):
        np.save(os.path.join(dir_save_train, f'label_{i:03d}.npy'), img_label)
        np.save(os.path.join(dir_save_train, f'input_{i:03d}.npy'), img_input)

    # val 데이터를 저장
    for i, (img_input, img_label) in enumerate(zip(val_images, val_labels)):
        np.save(os.path.join(dir_save_val, f'label_{i:03d}.npy'), img_label)
        np.save(os.path.join(dir_save_val, f'input_{i:03d}.npy'), img_input)

    # test 데이터를 저장
    for i, (img_input, img_label) in enumerate(zip(test_images, test_labels)):
        np.save(os.path.join(dir_save_test, f'label_{i:03d}.npy'), img_label)
        np.save(os.path.join(dir_save_test, f'input_{i:03d}.npy'), img_input)


if __name__ == '__main__':
    data_dir=".\\dataset\\"
    # download_kaggle_dset(data_dir)
    data_preprocessing(data_dir)