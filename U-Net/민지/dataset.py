import os

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
# https://hyunlee103.tistory.com/57
import torchvision.transforms as transforms
class DatasetForSeg(torch.utils.data.Dataset):

    def download_kaggle_dset(self, path):
        # Kaggle dataset 다운로드
        os.system(f"kaggle datasets download -d santhoshkumarv/dog-segmentation-dataset -p {path} --unzip")
        print("Dataset downloaded to:", path)

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        # 문자열 검사해서 'label'이 있으면 True
        # 문자열 검사해서 'input'이 있으면 True
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)


    # 여기가 데이터 load하는 파트
    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        inputs = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # normalize, 이미지는 0~255 값을 가지고 있어 이를 0~1사이로 scaling
        label = label / 255.0
        inputs = inputs / 255.0
        label = label.astype(np.float32)
        inputs = inputs.astype(np.float32)
        # 파이토치 인풋은 (batch, 채널, 행, 열)

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if inputs.ndim == 2:
            inputs = inputs[:, :, np.newaxis]
        input_canny = np.stack([cv2.Canny((inputs[..., i] * 255).astype(np.uint8), 100, 200) for i in range(3)], axis=0)
        data = {'input': inputs, 'input_canny': input_canny, 'label': label}
        input_canny = input_canny.transpose(1,2,0)

        if self.transform:
            transform_RGB, transform_gray = self.transform[0], self.transform[1]
            transformed_input = transform_RGB(inputs)
            transformed_label = transform_gray(label)
            transformed_canny = transform_RGB(input_canny)
            data = {'input': transformed_input, 'input_canny': transformed_canny, 'label': transformed_label}
        # transform에 할당된 class 들이 호출되면서 __call__ 함수 실행

        return data

    def show_image(self):
        print("### Number of classes:", self.__len__())
        print("### Number of samples:", self.__len__())

        random_index = np.random.randint(0, self.__len__())
        input_tensor, canny_tensor, label_tensor = self.__getitem__(random_index)['input'],self.__getitem__(random_index)['input_canny'], self.__getitem__(random_index)['label']
        print("### Shape of each image:", input_tensor.shape)

        input_numpy = input_tensor.permute(1, 2, 0).numpy()
        input_numpy = input_numpy * 0.5 + 0.5

        canny_numpy = canny_tensor.permute(1, 2, 0).numpy()
        canny_numpy = canny_numpy * 0.5 + 0.5

        label_numpy = label_tensor.squeeze().numpy()

        # To restore values from [0, 1] to [0, 255]
        input_numpy = (input_numpy * 255).astype(np.uint8)
        canny_numpy = (canny_numpy * 255).astype(np.uint8)
        # print("확인용: ", input_numpy.max(), input_numpy.min())
        label_numpy = (label_numpy).astype(np.uint8)


        # 이미지로 변환
        input_image = Image.fromarray(input_numpy)
        canny_image =  Image.fromarray(canny_numpy)
        label_image = Image.fromarray(label_numpy)

        fig, axes = plt.subplots(1, 3, figsize=(10, 5))

        axes[0].imshow(input_image)
        axes[0].set_title("Input")
        axes[1].imshow(label_image, cmap='gray')
        axes[1].set_title("Label")
        axes[2].imshow(canny_image, cmap='gray')
        axes[2].set_title("Canny")
        plt.suptitle("Data - Input / Label")  # 전체 제목 설정
        plt.show()
def data_transform():
    transform_RGB = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((512, 512)),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

    transform_gray = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((512, 512))
                                         ])

    return transform_RGB, transform_gray

if __name__ == '__main__':
    transform = data_transform()
    dataset = DatasetForSeg(data_dir='./dataset/train/', transform=transform)
    dataset.show_image()