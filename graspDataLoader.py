import torch
import torch.utils.data
import os
import numpy as np
import glob


class GraspDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, pattern='train', seed=42):
        self.seed = np.random.seed(seed)
        self.dataset_dir = dataset_dir

        self.dataPattern = pattern

        self.grasps_files = glob.glob(os.path.join(self.dataset_dir, 'tensor', 'poseTensor*.npz'))
        self.grasps_files.sort()

        self.img_files = glob.glob(os.path.join(self.dataset_dir, 'tensor', 'imgTensor*.npz'))
        self.img_files.sort()

        self.metrics_files = glob.glob(os.path.join(self.dataset_dir, 'tensor', 'metricTensor*.npz'))
        self.metrics_files.sort()

        self.imgTensor = np.load(self.img_files[0])['arr_0']
        self.poseTensor = np.load(self.grasps_files[0])['arr_0']
        self.metricTensor = np.load(self.metrics_files[0])['arr_0']

        for i in range(1, len(self.img_files)):
            self.imgTensor = np.concatenate((
                self.imgTensor, np.load(self.img_files[i])['arr_0']
            ))
            self.poseTensor = np.concatenate((
                self.poseTensor, np.load(self.grasps_files[i])['arr_0']
            ))
            self.metricTensor = np.concatenate((
                self.metricTensor, np.load(self.metrics_files[i])['arr_0']
            ))
        self.dataLen = self.imgTensor.shape[0]

        if self.dataPattern == 'train':
            self.imgTensor = self.imgTensor[0:int(0.9*self.dataLen), ...]
            self.poseTensor = self.poseTensor[0:int(0.9*self.dataLen), ...]
            self.metricTensor = self.metricTensor[0:int(0.9*self.dataLen), ...]
        if self.dataPattern == 'val':
            self.imgTensor = self.imgTensor[int(0.9 * self.dataLen):, ...]
            self.poseTensor = self.poseTensor[int(0.9 * self.dataLen):, ...]
            self.metricTensor = self.metricTensor[int(0.9 * self.dataLen):, ...]

    def __getitem__(self, item):

        img = self.imgTensor[item, ...]
        pose = self.poseTensor[item, ...]
        metric = self.metricTensor[item, ...]

        # print(img.shape)
        flippedFlag = np.random.rand(2)
        # if flippedFlag[0] > 0.5:
        #     img = img[::-1, :, :].copy()
        # if flippedFlag[1] > 0.5:
        #     img = img[:, ::-1, :].copy()

        return img, pose, metric

    def __len__(self):
        return self.metricTensor.shape[0]


if __name__ == '__main__':
    dataset = GraspDataset('/home/wangchuxuan/PycharmProjects/grasp/output')
    import torch.utils.data as D
    import time
    from matplotlib import pyplot as plt
    train_data = D.DataLoader(dataset, batch_size=1)
    print(train_data.__len__())
    num = 0
    t0 = time.time()
    for img, depth, metric in train_data:
        num += 1
        plt.imshow(img.squeeze(0).squeeze(2).cpu().detach().numpy())
        plt.show()
        # if num > 4000:
        #     plt.imshow(img.squeeze(0).squeeze(0).cpu().detach().numpy())
        #     plt.show()
    # print(num)




