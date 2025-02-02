import os
import wandb
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import UNet
from dataset import DatasetForSeg, data_transform



def calculate_IOU(groundtruth_mask, pred_mask):
    # PyTorch 텐서인 경우 numpy 변환
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().numpy()
    if isinstance(groundtruth_mask, torch.Tensor):
        groundtruth_mask = groundtruth_mask.detach().numpy()

    # IOU 계산
    intersect = np.sum(pred_mask * groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect

    # 예외 처리 (0으로 나누는 경우 방지)
    if union == 0:
        return 1.0 if intersect == 0 else 0.0

    iou = intersect / union
    return round(iou, 3)


def train_model(setting_config: dict):
    # setting
    batch_size = setting_config['batch_size']
    learning_rate = setting_config['learning_rate']
    num_epoch = setting_config['num_epoch']
    device = setting_config['device']

    # data
    data_dir = "./dataset/"
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    transform = data_transform()
    train_set = DatasetForSeg(data_dir=train_dir, transform=transform)
    test_set = DatasetForSeg(data_dir=test_dir, transform=transform)

    # DataLoader : 미니배치(batch) 단위로 데이터를 제공
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Trainer
    model = UNet(in_channel=3, out_channel=1).to(device)
    loss_func = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99)

    ## wandb
    wandb.init(project="unet-dog-training", name="experiment", config={
        "epochs": num_epoch,
        "batch_size": train_loader.batch_size,
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": optimizer.param_groups[0]['lr'],
    })

    loss_arr = []
    for i in tqdm(range(num_epoch), total=num_epoch, desc='training...'):
        for batch, data in enumerate(train_loader):
            model.train()
            label = data['label'].to(device)
            inputs = data['input'].to(device)
            output = model(inputs)  # forward
            loss = loss_func(output, label)

            pred_mask = (output > 0.5).float()
            iou = calculate_IOU(label, pred_mask)
            print("\n##### IOU : ", iou)
            wandb.log({"IOU": iou, "epoch": i})

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_arr.append(loss.item())
            wandb.log({"train_loss": loss.item(), "epoch": i})

            model.eval()
            with torch.no_grad():
                # --- Segmentation 이미지 로깅 ---
                # 단일 데이터이므로 앞에 배치 차원 추가 unsqueeze(0)
                label_val = test_set[0]['label'].unsqueeze(0).to(device)
                inputs_val = test_set[0]['input'].unsqueeze(0).to(device)
                output_val = model(inputs_val)

                pred_mask = output_val.squeeze(1)  # (batch, H, W) -> 예측된 segmentation mask
                label_mask = label_val.squeeze(1)  # GT mask (batch, H, W)

                pred_mask_np = pred_mask[0].cpu().numpy()
                print('shape of pred_mask: ', pred_mask_np.shape)
                label_mask_np = label_mask[0].cpu().numpy()
                print('shape of label_mask_np: ', label_mask_np.shape)

                wandb.log({
                    "Predicted Mask": wandb.Image(pred_mask_np, caption="Prediction"),
                    "Ground Truth": wandb.Image(label_mask_np, caption="Ground Truth"),
                })

        if i%10 == 0:
            print(f'Epoch {i}  Loss : ', loss.item())
            loss_arr.append(loss.cpu().detach().numpy())

    wandb.finish()


    # 학습 완료된 모델 저장
    torch.save(model.state_dict(), setting_config['save_model_path'])

if __name__ == '__main__':
    setting_config = {
        "batch_size": 16,
        "learning_rate": 0.001,
        "num_epoch": 10,
        "device": torch.device("cuda" if torch.cuda.is_available() else 'cpu'),
        "save_model_path": "./model/unet_dog.pth"
    }
    train_model(setting_config)
