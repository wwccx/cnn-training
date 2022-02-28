import torch
from graspDataLoader import GraspDataset
from network import GQCNN
import torch.utils.data as D
import datetime
import os
import logging
import time
import numpy as np
logging.basicConfig(level=logging.INFO)


class gqTrain:
    def __init__(self, dataDir='/home/wangchuxuan/PycharmProjects/grasp/output',
                 saveDir='/home/wangchuxuan/PycharmProjects/grasp/output/finetune_offpolicy'):

        self.device = torch.device("cuda:0")

        self.dataDir = dataDir
        self.saveDir = os.path.join(saveDir, datetime.datetime.now().strftime('%y_%m_%d_%H:%M'))
        logging.info('save path:'+self.saveDir)
        if not os.path.exists(self.saveDir):
            os.makedirs(self.saveDir)
        self.trainDataset = GraspDataset(self.dataDir)
        self.valDataset = GraspDataset(self.dataDir, pattern='val')

        self.trainDataLoader = D.DataLoader(self.trainDataset, batch_size=16, num_workers=1, shuffle=True)
        self.trainDataLen = self.trainDataLoader.__len__()
        # print(self.trainDataLen)

        self.valDataLoader = D.DataLoader(self.valDataset, batch_size=4, num_workers=1, shuffle=True)
        self.valDataLen = self.valDataLoader.__len__()
        # print(self.valDataLen)
        logging.info('data set loaded')
        self.network = GQCNN().to(self.device)


        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.01, momentum=0.9,
                                         weight_decay=0.005)
        self.lrScheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.93)
        self.currentEpoch = 0

        self.img_std = torch.tensor(np.load('im_std.npy')).cuda().float()
        self.img_mean = torch.tensor(np.load('im_mean.npy')).cuda().float()
        self.pose_std = torch.tensor(np.load('pose_std.npy')).cuda().float()
        self.pose_mean = torch.tensor(np.load('pose_mean.npy')).cuda().float()

    def train(self, log_frequency=4):
        self.network.train()
        batchIdx = 0
        t0 = time.time()
        success_pre_positive = torch.tensor(0.1).cuda()
        success_pre_negative = torch.tensor(0.1).cuda()
        total_pre_positive = torch.tensor(0.1).cuda()
        total_pre_negative = torch.tensor(0.1).cuda()

        for img, depth, metric in self.trainDataLoader:
            # img -= depth.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            img = img.permute((0, 3, 1, 2)).to(self.device).float()
            depth = depth.to(self.device)

            poseTensorNorm = (depth - self.pose_mean) / self.pose_std
            img = (img - self.img_mean) / self.img_std

            img -= poseTensorNorm.unsqueeze(1).unsqueeze(1)

            metric = metric.to(self.device).long()
            loss, label_pred = self.network.compute_loss(img, metric)

            judge_tensor = (label_pred == metric)
            total_pre_positive += torch.sum(metric == 1)
            total_pre_negative += torch.sum(metric == 0)
            success_pre_negative += torch.sum(judge_tensor[torch.where(metric == 0)])
            success_pre_positive += torch.sum(judge_tensor[torch.where(metric == 1)])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batchIdx % log_frequency == 0:
                logging.info('current Epoch:{}--{:.4f}%, loss:{:.6e}, batch time:{:.3f}s, epoch remain:{:.2f}min'

                             .format(self.currentEpoch, 100*batchIdx/self.trainDataLen,
                                     loss.item(),
                                     (time.time()-t0)/log_frequency,
                                     (time.time()-t0)/log_frequency*(self.trainDataLen - batchIdx)/60
                                     )
                             + ', NG {:.3f}%, PG {:.3f}%'
                             .format(
                                    100 * success_pre_negative / total_pre_negative,
                                    100 * success_pre_positive / total_pre_positive)
                             )

                # print(self.network(img))
                t0 = time.time()
            batchIdx += 1



    def validate(self, maxBatch=3000):
        self.network.eval()
        accuracy = 0
        valBatchIdx = 0
        success_pre_positive = torch.tensor(0.1).cuda()
        success_pre_negative = torch.tensor(0.1).cuda()
        total_pre_positive = torch.tensor(0.1).cuda()
        total_pre_negative = torch.tensor(0.1).cuda()

        for img, depth, metric in self.valDataLoader:
            img = img.permute((0, 3, 1, 2)).to(self.device).float()
            metric = metric.to(self.device).long()
            depth = depth.to(self.device).float()

            poseTensorNorm = (depth - self.pose_mean) / self.pose_std
            img = (img - self.img_mean) / self.img_std


            img -= poseTensorNorm.unsqueeze(1).unsqueeze(1)

            label_pred = self.network.get_label(img)
            judge_tensor = (label_pred == metric)

            total_pre_positive += torch.sum(metric == 1)
            total_pre_negative += torch.sum(metric == 0)
            success_pre_negative += torch.sum(judge_tensor[torch.where(metric == 0)])
            success_pre_positive += torch.sum(judge_tensor[torch.where(metric == 1)])

            if valBatchIdx % 2 == 0:
                logging.info('evalutating:{:.2f}%, negative grasp pre{:.3f}%, positive grasp pre{:.3f}%'
                             .format(100*valBatchIdx/maxBatch,
                                     100*success_pre_negative/total_pre_negative,
                                     100*success_pre_positive/total_pre_positive)
                             )

            valBatchIdx += 1
            if valBatchIdx > maxBatch:
                # torch.nn.CrossEntropyLoss()
                break

        return (success_pre_negative/total_pre_negative,
                success_pre_positive/total_pre_positive)

    def save(self, epoch, accuracy):
        dt = datetime.datetime.now().strftime('%H%M')
        save_path = os.path.join(self.saveDir, dt+'_epoch{}_Nacc{:.5f}_Pacc{:.5f}.pth'.format(epoch, accuracy[0],
                                                                                              accuracy[1]))
        torch.save(self.network.state_dict(), save_path)
        logging.info('save to '+save_path)

    def run(self, epoch=40000):
        for i in range(epoch):
            self.train()
            if i % 10 == 0:
                accuracy = self.validate()
                self.save(self.currentEpoch, accuracy)
                self.lrScheduler.step()
            self.currentEpoch += 1


if __name__ == '__main__':
    gqTrain = gqTrain()
    gqTrain.run()






