import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import SRCNN
from datasets import TrainDataset, TestDataset
from utils import AverageMeter, calc_psnr



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    # 학습 데이터 경로
    parser.add_argument('--train-dir', type=str, default='./db/train')
    # 테스트 데이터 경로
    parser.add_argument('--test-dir', type=str, default='./db/test')
    # 모델 파일 저장 경로
    parser.add_argument('--outputs-dir', type=str, default='./outputs')
    # 학습률
    parser.add_argument('--lr', type=float, default=1e-4)
    # 미니 배치 크기
    parser.add_argument('--batch-size', type=int, default=16)
    # 에포크 수 설정
    parser.add_argument('--num-epochs', type=int, default=400)
    # 아래의 3개 argument는 변경 불필요
    # 4배로 업스케일
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()


    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    #models.py의 모델 수정할 것
    model = SRCNN().to(device)

    #MSE 목적함수 사용
    criterion = nn.MSELoss()
    #Adam optimizer 사용
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = TrainDataset(args.train_dir, is_train=1, scale=args.scale)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True,
                                  drop_last=True)
    #eval_dataset = TrainDataset(args.test_dir, is_train=0, scale=args.scale)
    eval_dataset = TestDataset(args.test_dir, scale=args.scale)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False)


    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):

        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                # 입력과 정답값
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 예측값
                preds = model(inputs)

                # 예측값과 정답값 사이의 목적함수 (손실함수) 계산
                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                # 그레디언트 값을 0으로 초기화
                optimizer.zero_grad()
                # 오차역전파 수행
                loss.backward()
                # optimizer를 다음 스텝으로
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
        torch.save(model.state_dict(), "{}/epoch_{}.pth".format(args.outputs_dir, epoch))

        model.eval()
        epoch_psnr = AverageMeter()

        if (epoch + 1) % 5 == 0:
            epoch_psnr = torch.zeros(len(eval_dataloader))
            n = -1
            for data in eval_dataloader:
                n += 1
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    preds = model(inputs).clamp(0.0, 1.0)
                epoch_psnr[n] = calc_psnr(preds, labels)

            epoch_psnr_avg = torch.mean(epoch_psnr)
            print('eval psnr: {:.2f}'.format(epoch_psnr_avg ))

            if epoch_psnr_avg > best_psnr:
                best_epoch = epoch
                best_psnr = epoch_psnr_avg
                best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    #최적의 매개변수가 best.pth에 저장
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
