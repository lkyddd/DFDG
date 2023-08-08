import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets as vision_datasets
from lightfed.tools.funcs import save_pkl
from PIL import Image

from pytorch_cinic.dataset import CINIC10

from torchvision import models, utils
import sys
from torchvision.datasets import ImageFolder, DatasetFolder



class TransDataset(Dataset):
    def __init__(self, dataset, transform) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return self.transform(img), label

    def __len__(self):
        return len(self.dataset)


class ListDataset(Dataset):
    def __init__(self, data_list) -> None:
        super().__init__()
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.trsf = trsf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.trsf(pil_loader(self.images[idx]))
        label = self.labels[idx]

        return image, label





class DataDistributer:
    def __init__(self, args, dataset_dir=None, cache_dir=None):
        if dataset_dir is None:
            dataset_dir = os.path.abspath(os.path.join(__file__, "../../../../dataset"))

        if cache_dir is None:
            cache_dir = f"{dataset_dir}/cache_data"

        self.dataset_dir = dataset_dir
        self.cache_dir = cache_dir
        self.args = args
        self.data_set = args.data_set
        self.client_num = args.client_num
        self.batch_size = args.batch_size

        self.class_num = None
        self.x_shape = None  # 每个样本x的shape
        self.client_train_dataloaders = []
        self.client_test_dataloaders = []
        self.train_dataloaders = None
        self.test_dataloaders = None
        self.client_label_list = None
        self.client_label_count = None

        _dataset_load_func = getattr(self, f'_load_{args.data_set.replace("-","_")}')
        _dataset_load_func()


    def get_client_train_dataloader(self, client_id):
        return self.client_train_dataloaders[client_id]

    def get_client_test_dataloader(self, client_id):
        return self.client_test_dataloaders[client_id]

    def get_client_label_list(self, client_id):
        return self.client_label_list[client_id]

    def get_client_label_count(self):
        return self.client_label_count

    def get_train_dataloader(self):
        return self.train_dataloaders

    def get_test_dataloader(self):
        return self.test_dataloaders

    def _load_MNIST(self):
        self.class_num = 10
        self.x_shape = (1, 32, 32)

        # try:
        #     if self.args.generator_model_type in ['BEGAN', 'CGAN']:
        #         transform = transforms.Compose([transforms.Resize(32), 
        #                                         transforms.ToTensor(),
        #                                         transforms.Normalize((0.1307,), (0.3081,))])
        #     else:
        #         transform = transforms.Compose([transforms.ToTensor(),
        #                                         transforms.Normalize((0.1307,), (0.3081,))])
        # except:
        #     transform = transforms.Compose([transforms.ToTensor(),
        #                                         transforms.Normalize((0.1307,), (0.3081,))])

        transform = transforms.Compose([transforms.Resize(32), 
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = vision_datasets.MNIST(root=f"{self.dataset_dir}/MNIST", train=True, download=True, transform=transform)
        test_dataset = vision_datasets.MNIST(root=f"{self.dataset_dir}/MNIST", train=False, download=True, transform=transform)
        ###train data
        if len(train_dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_train = True
        else:
            drop_last_train = False
        # test data
        if len(test_dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_test = True
        else:
            drop_last_test = False
        self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size,
                                            drop_last=drop_last_train, shuffle=True)
        self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size,
                                           drop_last=drop_last_test, shuffle=True)

        if self.args.data_partition_mode == 'None':
            return

        client_train_datasets, client_test_datasets, self.client_label_list, self.client_label_count = self._split_dataset(train_dataset, test_dataset)
        # client_train_datasets = self._split_dataset(train_dataset, test_dataset)
        self.client_train_dataloaders = []
        self.client_test_dataloaders = []
        for client_id in range(self.client_num):
            _train_dataset = client_train_datasets[client_id]
            _test_dataset = client_test_datasets[client_id]
            if self.args.batch_size > 0:
                if len(_train_dataset) >= self.args.batch_size:
                    if len(_train_dataset.data_list) % self.args.batch_size == 1:
                        drop_last_train = True
                    else:
                        drop_last_train = False
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, drop_last=drop_last_train, shuffle=False)
                elif len(_train_dataset) < self.args.batch_size:
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)

                _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)

            elif self.args.batch_size == 0:
                _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)
                _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)
            self.client_train_dataloaders.append(_train_dataloader)
            self.client_test_dataloaders.append(_test_dataloader)

    def _load_FMNIST(self):
        self.class_num = 10
        self.x_shape = (1, 32, 32)
        if not os.path.exists(f"{self.dataset_dir}/FMNIST/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth"):
            transform = transforms.Compose([transforms.Resize(32), 
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])

            # transform = transforms.Compose([transforms.ToTensor()])

            # raw_train_dataset = vision_datasets.FashionMNIST(root=f"{self.dataset_dir}/FMNIST", train=True, download=True)
            # raw_test_dataset = vision_datasets.FashionMNIST(root=f"{self.dataset_dir}/FMNIST", train=False, download=True)

            # self.raw_train_data = raw_train_dataset
            # self.raw_test_data = raw_test_dataset

            train_dataset = vision_datasets.FashionMNIST(root=f"{self.dataset_dir}/FMNIST", train=True, download=True,
                                                        transform=transform)
            test_dataset = vision_datasets.FashionMNIST(root=f"{self.dataset_dir}/FMNIST", train=False, download=True, transform=transform)

            ###train data
            if len(train_dataset.targets) % self.args.eval_batch_size == 1:
                drop_last_train = True
            else:
                drop_last_train = False
            # test data
            if len(test_dataset.targets) % self.args.eval_batch_size == 1:
                drop_last_test = True
            else:
                drop_last_test = False
            self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size,
                                                drop_last=drop_last_train, shuffle=False)
            self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size,
                                            drop_last=drop_last_test, shuffle=False)
            
            torch.save((self.train_dataloaders, self.test_dataloaders), \
                       f"{self.dataset_dir}/FMNIST/full_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")

            if self.args.data_partition_mode == 'None':
                return

            client_train_datasets, client_test_datasets, self.client_label_list, self.client_label_count = self._split_dataset(train_dataset, test_dataset)
            # client_train_datasets = self._split_dataset(train_dataset, test_dataset)
            for client_id in range(self.client_num):
                _train_dataset = client_train_datasets[client_id]
                _test_dataset = client_test_datasets[client_id]
                if self.args.batch_size > 0:
                    if len(_train_dataset) >= self.args.batch_size:
                        if len(_train_dataset.data_list) % self.args.batch_size == 1:
                            drop_last_train = True
                        else:
                            drop_last_train = False
                        _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, drop_last=drop_last_train, shuffle=False)
                    elif len(_train_dataset) < self.args.batch_size:
                        _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)

                    _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=self.args.batch_size, shuffle=False)

                elif self.args.batch_size == 0:
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)
                    _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)
                self.client_train_dataloaders.append(_train_dataloader)
                self.client_test_dataloaders.append(_test_dataloader)
            torch.save((self.client_train_dataloaders, self.client_test_dataloaders, self.client_label_list, self.client_test_label_list, self.client_label_count), \
                       f"{self.dataset_dir}/FMNIST/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")
        else:
            self.train_dataloaders, self.test_dataloaders = torch.load(f"{self.dataset_dir}/FMNIST/full_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")
            if self.args.data_partition_mode == 'None':
                return
            self.client_train_dataloaders ,self.client_test_dataloaders, self.client_label_list, self.client_test_label_list, self.client_label_count = torch.load(f"{self.dataset_dir}/FMNIST/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")



    def _load_CIFAR_10(self):
        self.class_num = 10
        self.x_shape = (3, 32, 32)

        if not os.path.exists(f"{self.dataset_dir}/CIFAR-10/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth"):
            # 只开RandomCrop和RandomHorizontalFlip是标准的CIFA-10数据扩展
            train_transform = transforms.Compose([
                # transforms.RandomRotation(degrees=10),  # 随机旋转
                transforms.RandomCrop(size=32, padding=4),  # 填充后裁剪
                transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
                transforms.ToTensor(),
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  # 颜色变化。亮度
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

            raw_train_dataset = vision_datasets.CIFAR10(root=f"{self.dataset_dir}/CIFAR-10", train=True, download=True)
            raw_test_dataset = vision_datasets.CIFAR10(root=f"{self.dataset_dir}/CIFAR-10", train=False, download=True)

            train_dataset = TransDataset(raw_train_dataset, train_transform)
            test_dataset = TransDataset(raw_test_dataset, test_transform)
            ###train data
            if len(train_dataset.dataset.targets) % self.args.eval_batch_size == 1:
                drop_last_train = True
            else:
                drop_last_train = False
            # test data
            if len(test_dataset.dataset.targets) % self.args.eval_batch_size == 1:
                drop_last_test = True
            else:
                drop_last_test = False
            self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size,
                                                drop_last=drop_last_train, shuffle=True)
            self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size,
                                            drop_last=drop_last_test, shuffle=True)
            
            torch.save((self.train_dataloaders, self.test_dataloaders), \
                       f"{self.dataset_dir}/CIFAR-10/full_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")

            if self.args.data_partition_mode == 'None':
                return

            raw_client_train_datasets, raw_client_test_datasets, self.client_label_list, self.client_label_count = self._split_dataset(raw_train_dataset, raw_test_dataset)
            # raw_client_train_datasets = self._split_dataset(raw_train_dataset, raw_test_dataset)
            self.client_train_dataloaders = []
            self.client_test_dataloaders = []
            for client_id in range(self.client_num):
                _raw_train_dataset = raw_client_train_datasets[client_id]
                _raw_test_dataset = raw_client_test_datasets[client_id]
                _train_dataset = TransDataset(_raw_train_dataset, train_transform)
                _test_dataset = TransDataset(_raw_test_dataset, test_transform)
                if self.args.batch_size > 0:
                    if len(_train_dataset) >= self.args.batch_size:
                        if len(_train_dataset.dataset.data_list) % self.args.batch_size == 1:
                            drop_last_train = True
                        else:
                            drop_last_train = False
                        _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, drop_last=drop_last_train, shuffle=False)
                    elif len(_train_dataset) < self.args.batch_size:
                        _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)

                    _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=self.args.batch_size, shuffle=False)

                elif self.args.batch_size == 0:
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)
                    _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)

                self.client_train_dataloaders.append(_train_dataloader)
                self.client_test_dataloaders.append(_test_dataloader)
            torch.save((self.client_train_dataloaders, self.client_test_dataloaders, self.client_label_list, self.client_test_label_list, self.client_label_count), \
                       f"{self.dataset_dir}/CIFAR-10/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")
        else:
            self.train_dataloaders, self.test_dataloaders = torch.load(f"{self.dataset_dir}/CIFAR-10/full_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")
            if self.args.data_partition_mode == 'None':
                return
            self.client_train_dataloaders ,self.client_test_dataloaders, self.client_label_list, self.client_test_label_list, self.client_label_count = torch.load(f"{self.dataset_dir}/CIFAR-10/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")

    def _load_CINIC_10(self):
        self.class_num = 10
        self.x_shape = (3, 32, 32)

        if not os.path.exists(f"{self.dataset_dir}/CINIC-10/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth"):
            # 只开RandomCrop和RandomHorizontalFlip是标准的CIFA-10数据扩展
            mean=[0.47889522, 0.47227842, 0.43047404]
            std=[0.47889522, 0.47227842, 0.43047404]
            train_transform = transforms.Compose([
                # transforms.RandomRotation(degrees=10),  # 随机旋转
                transforms.RandomCrop(size=32, padding=4),  # 填充后裁剪
                transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
                transforms.ToTensor(),
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  # 颜色变化。亮度
                transforms.Normalize(mean=mean, std=std)])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])

            raw_train_dataset = CINIC10(root=f"{self.dataset_dir}/CINIC-10", partition='train', download=True)
            raw_test_dataset = CINIC10(root=f"{self.dataset_dir}/CINIC-10", partition='test', download=True)

            train_dataset = TransDataset(raw_train_dataset, train_transform)
            test_dataset = TransDataset(raw_test_dataset, test_transform)
            ###train data
            if len(train_dataset.dataset) % self.args.eval_batch_size == 1:
                drop_last_train = True
            else:
                drop_last_train = False
            # test data
            if len(test_dataset.dataset) % self.args.eval_batch_size == 1:
                drop_last_test = True
            else:
                drop_last_test = False
            self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size,
                                                drop_last=drop_last_train, shuffle=True)
            self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size,
                                            drop_last=drop_last_test, shuffle=True)
            
            torch.save((self.train_dataloaders, self.test_dataloaders), \
                       f"{self.dataset_dir}/CINIC-10/full_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")

            if self.args.data_partition_mode == 'None':
                return

            raw_client_train_datasets, raw_client_test_datasets, self.client_label_list, self.client_label_count = self._split_dataset(raw_train_dataset, raw_test_dataset)
            # raw_client_train_datasets = self._split_dataset(raw_train_dataset, raw_test_dataset)
            self.client_train_dataloaders = []
            self.client_test_dataloaders = []
            for client_id in range(self.client_num):
                _raw_train_dataset = raw_client_train_datasets[client_id]
                _raw_test_dataset = raw_client_test_datasets[client_id]
                _train_dataset = TransDataset(_raw_train_dataset, train_transform)
                _test_dataset = TransDataset(_raw_test_dataset, test_transform)
                if self.args.batch_size > 0:
                    if len(_train_dataset) >= self.args.batch_size:
                        if len(_train_dataset.dataset.data_list) % self.args.batch_size == 1:
                            drop_last_train = True
                        else:
                            drop_last_train = False
                        _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, drop_last=drop_last_train, shuffle=False)
                    elif len(_train_dataset) < self.args.batch_size:
                        _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)

                    _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=self.args.batch_size, shuffle=False)

                elif self.args.batch_size == 0:
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)
                    _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)

                self.client_train_dataloaders.append(_train_dataloader)
                self.client_test_dataloaders.append(_test_dataloader)
            torch.save((self.client_train_dataloaders, self.client_test_dataloaders, self.client_label_list, self.client_test_label_list, self.client_label_count), \
                       f"{self.dataset_dir}/CINIC-10/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")
        else:
            self.train_dataloaders, self.test_dataloaders = torch.load(f"{self.dataset_dir}/CINIC-10/full_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")
            if self.args.data_partition_mode == 'None':
                return
            self.client_train_dataloaders ,self.client_test_dataloaders, self.client_label_list, self.client_test_label_list, self.client_label_count = torch.load(f"{self.dataset_dir}/CINIC-10/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")

    def _load_CIFAR_100(self):
        self.class_num = 100
        self.x_shape = (3, 32, 32)
        if not os.path.exists(f"{self.dataset_dir}/CIFAR-100/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth"):
            mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
            std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

            train_transform = transforms.Compose([
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

            raw_train_dataset = vision_datasets.CIFAR100(root=f"{self.dataset_dir}/CIFAR-100", train=True, download=True)
            raw_test_dataset = vision_datasets.CIFAR100(root=f"{self.dataset_dir}/CIFAR-100", train=False, download=True)

            train_dataset = TransDataset(raw_train_dataset, train_transform)
            test_dataset = TransDataset(raw_test_dataset, test_transform)
            ###train data
            if len(train_dataset.dataset.targets) % self.args.eval_batch_size == 1:
                drop_last_train = True
            else:
                drop_last_train = False
            #test data
            if len(test_dataset.dataset.targets) % self.args.eval_batch_size == 1:
                drop_last_test = True
            else:
                drop_last_test = False
            self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size, drop_last=drop_last_train, shuffle=True)
            self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size, drop_last=drop_last_test, shuffle=True)
            torch.save((self.train_dataloaders, self.test_dataloaders), \
                       f"{self.dataset_dir}/CIFAR-100/full_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")


            if self.args.data_partition_mode == 'None':
                return

            raw_client_train_datasets, raw_client_test_datasets, self.client_label_list, self.client_label_count = self._split_dataset(raw_train_dataset, raw_test_dataset)
            # raw_client_train_datasets  = self._split_dataset(raw_train_dataset, raw_test_dataset)
            self.client_train_dataloaders = []
            self.client_test_dataloaders = []
            for client_id in range(self.client_num):
                _raw_train_dataset = raw_client_train_datasets[client_id]
                _raw_test_dataset = raw_client_test_datasets[client_id]
                _train_dataset = TransDataset(_raw_train_dataset, train_transform)
                _test_dataset = TransDataset(_raw_test_dataset, test_transform)
                if self.args.batch_size > 0:
                    if len(_train_dataset) >= self.args.batch_size:
                        if len(_train_dataset.dataset.data_list) % self.args.batch_size == 1:
                            drop_last_train = True
                        else:
                            drop_last_train = False
                        _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, drop_last=drop_last_train, shuffle=False)
                    elif len(_train_dataset) < self.args.batch_size:
                        _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)

                    _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=self.args.batch_size, shuffle=False)
                    # print(len(_test_dataset))

                elif self.args.batch_size == 0:
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)
                    _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)

                self.client_train_dataloaders.append(_train_dataloader)
                self.client_test_dataloaders.append(_test_dataloader)
            torch.save((self.client_train_dataloaders, self.client_test_dataloaders, self.client_label_list, self.client_test_label_list, self.client_label_count), \
                       f"{self.dataset_dir}/CIFAR-100/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")
        else:
            self.train_dataloaders, self.test_dataloaders = torch.load(f"{self.dataset_dir}/CIFAR-100/full_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")
            if self.args.data_partition_mode == 'None':
                return
            self.client_train_dataloaders ,self.client_test_dataloaders, self.client_label_list, self.client_test_label_list, self.client_label_count = torch.load(f"{self.dataset_dir}/CIFAR-100/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")

    
    def _load_SVHN(self):
        self.class_num = 10
        self.x_shape = (3, 32, 32)
        if not os.path.exists(f"{self.dataset_dir}/SVHN/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth"):
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

            train_transform = transforms.Compose([
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
            
            raw_train_dataset = vision_datasets.SVHN(root=f"{self.dataset_dir}/SVHN", split='train', download=True)
            raw_test_dataset = vision_datasets.SVHN(root=f"{self.dataset_dir}/SVHN", split='test', download=True)

            train_dataset = TransDataset(raw_train_dataset, train_transform)
            test_dataset = TransDataset(raw_test_dataset, test_transform)
            ###train data
            if len(train_dataset.dataset.labels) % self.args.eval_batch_size == 1:
                drop_last_train = True
            else:
                drop_last_train = False
            #test data
            if len(test_dataset.dataset.labels) % self.args.eval_batch_size == 1:
                drop_last_test = True
            else:
                drop_last_test = False
            self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size, drop_last=drop_last_train, shuffle=True)
            self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size, drop_last=drop_last_test, shuffle=True)
            torch.save((self.train_dataloaders, self.test_dataloaders), \
                       f"{self.dataset_dir}/SVHN/full_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")


            if self.args.data_partition_mode == 'None':
                return

            raw_client_train_datasets, raw_client_test_datasets, self.client_label_list, self.client_label_count = self._split_dataset(raw_train_dataset, raw_test_dataset)
            # raw_client_train_datasets  = self._split_dataset(raw_train_dataset, raw_test_dataset)
            self.client_train_dataloaders = []
            self.client_test_dataloaders = []
            for client_id in range(self.client_num):
                _raw_train_dataset = raw_client_train_datasets[client_id]
                _raw_test_dataset = raw_client_test_datasets[client_id]
                _train_dataset = TransDataset(_raw_train_dataset, train_transform)
                _test_dataset = TransDataset(_raw_test_dataset, test_transform)
                if self.args.batch_size > 0:
                    if len(_train_dataset) >= self.args.batch_size:
                        if len(_train_dataset.dataset.data_list) % self.args.batch_size == 1:
                            drop_last_train = True
                        else:
                            drop_last_train = False
                        _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, drop_last=drop_last_train, shuffle=False)
                    elif len(_train_dataset) < self.args.batch_size:
                        _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)

                    _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=self.args.batch_size, shuffle=False)

                elif self.args.batch_size == 0:
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)
                    _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)

                self.client_train_dataloaders.append(_train_dataloader)
                self.client_test_dataloaders.append(_test_dataloader)
            torch.save((self.client_train_dataloaders, self.client_test_dataloaders, self.client_label_list, self.client_test_label_list, self.client_label_count), \
                       f"{self.dataset_dir}/SVHN/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")
        else:
            self.train_dataloaders, self.test_dataloaders = torch.load(f"{self.dataset_dir}/SVHN/full_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")
            if self.args.data_partition_mode == 'None':
                return
            self.client_train_dataloaders ,self.client_test_dataloaders, self.client_label_list, self.client_test_label_list, self.client_label_count = torch.load(f"{self.dataset_dir}/SVHN/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")

    
    def _load_FOOD101(self):
        self.class_num = 101
        self.x_shape = (3, 64, 64)
        
        if not os.path.exists(f"{self.dataset_dir}/Food101/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth"):
            mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
            std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
            train_transform = transforms.Compose([
                transforms.Resize([64, 64]), 
                # transforms.RandomCrop(size=512, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

            test_transform = transforms.Compose([
                transforms.Resize([64, 64]), 
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
            
            # raw_train_dataset = vision_datasets.Food101(root=f"{self.dataset_dir}/Food101", split='train', download=True, transform=train_transform)
            # raw_test_dataset = vision_datasets.Food101(root=f"{self.dataset_dir}/Food101", split='test', download=True, transform=test_transform)

            # train_dataset = TransDataset(raw_train_dataset, train_transform)
            # test_dataset = TransDataset(raw_test_dataset, test_transform)

            train_dataset = vision_datasets.Food101(root=f"{self.dataset_dir}/Food101", split='train', download=True, transform=train_transform)
            test_dataset = vision_datasets.Food101(root=f"{self.dataset_dir}/Food101", split='test', download=True, transform=test_transform)
            ###train data
            if len(train_dataset._labels) % self.args.eval_batch_size == 1:
                drop_last_train = True
            else:
                drop_last_train = False
            #test data
            if len(test_dataset._labels) % self.args.eval_batch_size == 1:
                drop_last_test = True
            else:
                drop_last_test = False
            self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size, drop_last=drop_last_train, shuffle=True)
            self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size, drop_last=drop_last_test, shuffle=True)

            torch.save((self.train_dataloaders, self.test_dataloaders), \
                       f"{self.dataset_dir}/Food101/full_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")

            if self.args.data_partition_mode == 'None':
                return

            raw_client_train_datasets, raw_client_test_datasets, self.client_label_list, self.client_label_count = self._split_dataset(train_dataset, test_dataset)
            # raw_client_train_datasets  = self._split_dataset(raw_train_dataset, raw_test_dataset)
            self.client_train_dataloaders = []
            self.client_test_dataloaders = []
            for client_id in range(self.client_num):
                _train_dataset = raw_client_train_datasets[client_id]
                _test_dataset = raw_client_test_datasets[client_id]
                if self.args.batch_size > 0:
                    if len(_train_dataset) >= self.args.batch_size:
                        if len(_train_dataset.data_list) % self.args.batch_size == 1:
                            drop_last_train = True
                        else:
                            drop_last_train = False
                        _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, drop_last=drop_last_train, shuffle=False)
                    elif len(_train_dataset) < self.args.batch_size:
                        _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)

                    _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=self.args.batch_size, shuffle=False)

                elif self.args.batch_size == 0:
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)
                    _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)

                self.client_train_dataloaders.append(_train_dataloader)
                self.client_test_dataloaders.append(_test_dataloader)
            
            torch.save((self.client_train_dataloaders, self.client_test_dataloaders, self.client_label_list, self.client_test_label_list, self.client_label_count), \
                       f"{self.dataset_dir}/Food101/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")
        else:
            self.train_dataloaders, self.test_dataloaders = torch.load(f"{self.dataset_dir}/Food101/full_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")
            if self.args.data_partition_mode == 'None':
                return
            self.client_train_dataloaders ,self.client_test_dataloaders, self.client_label_list, self.client_test_label_list, self.client_label_count = torch.load(f"{self.dataset_dir}/Food101/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")
    
    
    def _load_GTSRB(self):
        self.class_num = 43
        self.x_shape = (3, 64, 64)
        
        if not os.path.exists(f"{self.dataset_dir}/GTSRB/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth"):
            mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
            std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
            train_transform = transforms.Compose([
                transforms.Resize([self.x_shape[1], self.x_shape[2]]),  # max 225 243
                # transforms.RandomCrop(size=512, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

            test_transform = transforms.Compose([
                transforms.Resize([self.x_shape[1], self.x_shape[2]]), # max 232 266 
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
            
            # raw_train_dataset = vision_datasets.Food101(root=f"{self.dataset_dir}/Food101", split='train', download=True, transform=train_transform)
            # raw_test_dataset = vision_datasets.Food101(root=f"{self.dataset_dir}/Food101", split='test', download=True, transform=test_transform)

            # train_dataset = TransDataset(raw_train_dataset, train_transform)
            # test_dataset = TransDataset(raw_test_dataset, test_transform)

            train_dataset = vision_datasets.GTSRB(root=f"{self.dataset_dir}/GTSRB", split='train', download=True, transform=train_transform)
            test_dataset = vision_datasets.GTSRB(root=f"{self.dataset_dir}/GTSRB", split='test', download=True, transform=test_transform)

            ###train data
            if len(train_dataset) % self.args.eval_batch_size == 1:
                drop_last_train = True
            else:
                drop_last_train = False
            #test data
            if len(test_dataset) % self.args.eval_batch_size == 1:
                drop_last_test = True
            else:
                drop_last_test = False
            self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size, drop_last=drop_last_train, shuffle=True)
            self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size, drop_last=drop_last_test, shuffle=True)

            torch.save((self.train_dataloaders, self.test_dataloaders), \
                       f"{self.dataset_dir}/GTSRB/full_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")

            if self.args.data_partition_mode == 'None':
                return

            raw_client_train_datasets, raw_client_test_datasets, self.client_label_list, self.client_label_count = self._split_dataset(train_dataset, test_dataset)
            # raw_client_train_datasets  = self._split_dataset(raw_train_dataset, raw_test_dataset)
            self.client_train_dataloaders = []
            self.client_test_dataloaders = []
            for client_id in range(self.client_num):
                _train_dataset = raw_client_train_datasets[client_id]
                _test_dataset = raw_client_test_datasets[client_id]
                if self.args.batch_size > 0:
                    if len(_train_dataset) >= self.args.batch_size:
                        if len(_train_dataset.data_list) % self.args.batch_size == 1:
                            drop_last_train = True
                        else:
                            drop_last_train = False
                        _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, drop_last=drop_last_train, shuffle=False)
                    elif len(_train_dataset) < self.args.batch_size:
                        _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)

                    _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=self.args.batch_size, shuffle=False)

                elif self.args.batch_size == 0:
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)
                    _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)

                self.client_train_dataloaders.append(_train_dataloader)
                self.client_test_dataloaders.append(_test_dataloader)
            
            torch.save((self.client_train_dataloaders, self.client_test_dataloaders, self.client_label_list, self.client_test_label_list, self.client_label_count), \
                       f"{self.dataset_dir}/GTSRB/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")
        else:
            self.train_dataloaders, self.test_dataloaders = torch.load(f"{self.dataset_dir}/GTSRB/full_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")
            if self.args.data_partition_mode == 'None':
                return
            self.client_train_dataloaders ,self.client_test_dataloaders, self.client_label_list, self.client_test_label_list, self.client_label_count = torch.load(f"{self.dataset_dir}/GTSRB/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")


    def _load_Tiny_Imagenet(self):
        self.class_num = 200
        self.x_shape = (3, 64, 64)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # mean = [0.5, 0.5, 0.5]
        # std = [0.5, 0.5, 0.5]
        data_dir = f"{self.dataset_dir}/tiny-imagenet-200"

        if not os.path.exists(f"{data_dir}/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth"):
            train_transform = transforms.Compose([
                transforms.RandomCrop(size=64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

            train_dir = data_dir + '/train/'
            test_dir = data_dir + '/val/'

            self.train_data, self.train_targets, self.test_data, self.test_targets = self.download_data(train_dir, test_dir)

            class_set = list(range(self.class_num))
            trainfolder = self.get_dataset(train_transform, index=class_set, train=True)
            testfolder = self.get_dataset(test_transform, index=class_set, train=False)

            ###train data
            if len(trainfolder) % self.args.eval_batch_size == 1: #len(train_dataset)
                drop_last_train = True
            else:
                drop_last_train = False
            #test data
            if len(testfolder) % self.args.eval_batch_size == 1:
                drop_last_test = True
            else:
                drop_last_test = False
            self.train_dataloaders = DataLoader(dataset=trainfolder, batch_size=self.args.eval_batch_size, drop_last=drop_last_train, shuffle=True)
            self.test_dataloaders = DataLoader(dataset=testfolder, batch_size=self.args.eval_batch_size, drop_last=drop_last_test, shuffle=True)

            torch.save((self.train_dataloaders, self.test_dataloaders), \
                       f"{data_dir}/full_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")

            if self.args.data_partition_mode == 'None':
                return

            raw_client_train_datasets, raw_client_test_datasets, self.client_label_list, self.client_label_count = self._split_dataset(trainfolder, testfolder)
            # raw_client_train_datasets = self._split_dataset(trainfolder, testfolder)
            self.client_train_dataloaders = []
            self.client_test_dataloaders = []
            for client_id in range(self.client_num):
                # _raw_train_dataset = raw_client_train_datasets[client_id]
                # # _raw_test_dataset = raw_client_test_datasets[client_id]
                # _train_dataset = TransDataset(_raw_train_dataset, train_transform)
                _train_dataset = raw_client_train_datasets[client_id]
                _test_dataset = raw_client_test_datasets[client_id]
                if self.args.batch_size > 0:
                    if len(_train_dataset) >= self.args.batch_size:
                        if len(_train_dataset) % self.args.batch_size == 1:
                            drop_last_train = True
                        else:
                            drop_last_train = False
                        _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, drop_last=drop_last_train, shuffle=False)
                    elif len(_train_dataset) < self.args.batch_size:
                        _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)

                    _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)

                elif self.args.batch_size == 0:
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)
                    _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)

                self.client_train_dataloaders.append(_train_dataloader)
                self.client_test_dataloaders.append(_test_dataloader)

            torch.save((self.client_train_dataloaders, self.client_test_dataloaders, self.client_label_list, self.client_test_label_list, self.client_label_count), \
                       f"{data_dir}/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")
        else:
            self.train_dataloaders, self.test_dataloaders = torch.load(f"{data_dir}/full_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")
            if self.args.data_partition_mode == 'None':
                return
            self.client_train_dataloaders ,self.client_test_dataloaders, self.client_label_list, self.client_test_label_list, self.client_label_count = torch.load(f"{data_dir}/client_{self.args.data_set}_{self.args.data_partition_mode}_{self.args.non_iid_alpha}_{self.args.client_num}_{self.args.seed}.pth")

    def download_data(self, train_dir, test_dir):
        train_dset = ImageFolder(train_dir)

        train_images = []
        train_labels = []
        for item in train_dset.imgs:
            train_images.append(item[0])
            train_labels.append(item[1])
        train_data, train_targets = np.array(train_images), np.array(train_labels)

        test_images = []
        test_labels = []
        _, class_to_idx = self.find_classes(train_dir)
        imgs_path = os.path.join(test_dir, 'images')
        imgs_annotations = os.path.join(test_dir, 'val_annotations.txt')
        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split('\t'), r.readlines())
        cls_map = {line_data[0]: line_data[1] for line_data in data_info}
        for imgname in sorted(os.listdir(imgs_path)):
            if cls_map[imgname] in sorted(class_to_idx.keys()):
                path = os.path.join(imgs_path, imgname)
                test_images.append(path)
                test_labels.append(class_to_idx[cls_map[imgname]])
        test_data, test_targets = np.array(test_images), np.array(test_labels)
        return train_data, train_targets, test_data, test_targets


    def find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def get_dataset(self, transform, index, train=True):
        if train:
            x, y = self.train_data, self.train_targets
        else:
            x, y = self.test_data, self.test_targets

        data, targets = [], []
        for idx in index:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            data.append(class_data)
            targets.append(class_targets)
        data, targets = np.concatenate(data), np.concatenate(targets)
        return DummyDataset(data, targets, transform)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]


    def _split_dataset(self, train_dataset, test_dataset):
        # partition_proportions中的第i行是第i个类别的数据被划分到各个客户端的比例
        if self.args.data_partition_mode == 'iid':
            partition_proportions = np.full(shape=(self.class_num, self.client_num), fill_value=1/self.client_num)
            client_train_datasets, client_label_list, client_label_count = self._split_dataset_iid(train_dataset, partition_proportions)

        elif self.args.data_partition_mode == 'non_iid_unbalanced':
            client_train_datasets, client_label_list, client_label_count = self._split_dataset_dirichlet_unbalanced(train_dataset, self.client_num, alpha=self.args.non_iid_alpha)

        elif self.args.data_partition_mode == 'non_iid_balanced':
            client_train_datasets, client_label_list, client_label_count = self._split_dataset_dirichlet_balanced(train_dataset, self.client_num, alpha=self.args.non_iid_alpha)
        else:
            raise Exception(f"unknow data_partition_mode:{self.args.data_partition_mode}")

        partition_proportions = np.full(shape=(self.class_num, self.client_num), fill_value=1 / self.client_num)
        client_test_datasets, self.client_test_label_list, _ = self._split_dataset_iid(test_dataset, partition_proportions)
        
        return client_train_datasets, client_test_datasets, client_label_list, client_label_count

    def _split_dataset_dirichlet_unbalanced(self, train_dataset, n_nets, alpha=0.01):

        if self.data_set in ['SVHN', 'Tiny-Imagenet']:
            y_train = train_dataset.labels
            K = len(set(y_train))
        elif self.data_set in ['FOOD101']:
            y_train = train_dataset._labels
            K = len(train_dataset.classes)
        elif self.data_set in ['GTSRB']:
            y_train = []
            for _, label in train_dataset._samples:
                y_train.append(label)
            K = self.class_num
        elif self.data_set in ['CINIC-10']:
            y_train = train_dataset.data.targets
            K = len(train_dataset.data.classes)
        else:
            y_train = train_dataset.targets
            K = len(train_dataset.class_to_idx)

        min_size = 0
        try:
            N = y_train.shape[0]
        except:
            y_train = np.array(y_train)
            N = y_train.shape[0]
        net_dataidx_map = {i: np.array([], dtype='int64') for i in range(n_nets)}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                pp_list = []
                for p, idx_j in zip(proportions, idx_batch):
                    pp = p * (len(idx_j) < N / n_nets)
                    pp_list.append(pp)
                proportions = np.array(pp_list)
                # proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch0 = []
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions)):
                    idx_batch0.append(idx_j + idx.tolist())
                idx_batch = idx_batch0
                # idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(n_nets):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]

        client_data_list = [[] for _ in range(self.client_num)]
        client_label_list = [[] for _ in range(self.client_num)]
        client_label_count = {j:{i: 0 for i in range(K)} for j in range(self.client_num)}
        for client_id, client_data, client_label in zip(net_dataidx_map, client_data_list, client_label_list):
            for id in net_dataidx_map[client_id]:
                tr_dat_id = train_dataset[id]
                client_data.append(tr_dat_id)
                client_label_count[client_id][tr_dat_id[1]] += 1
                if tr_dat_id[1] not in client_label:
                    client_label.append(tr_dat_id[1])
        # self.client_label_count = client_label_count
                
        client_datasets = []
        for client_data in client_data_list:
            # 打乱数据集
            np.random.shuffle(client_data)
            _dataset = ListDataset(client_data)
            client_datasets.append(_dataset)

        return client_datasets, client_label_list, client_label_count

    def _split_dataset_dirichlet_balanced(self, train_dataset, n_nets, alpha=0.5):

        if self.data_set in ['SVHN', 'Tiny-Imagenet']:
            y_train = train_dataset.labels
            K = len(set(y_train))
        elif self.data_set in ['FOOD101']:
            y_train = train_dataset._labels
            K = len(train_dataset.classes)
        elif self.data_set in ['GTSRB']:
            y_train = []
            for _, label in train_dataset._samples:
                y_train.append(label)
            K = self.class_num
        elif self.data_set in ['CINIC-10']:
            y_train = train_dataset.data.targets
            K = len(train_dataset.data.classes)
        else:
            y_train = train_dataset.targets
            K = len(train_dataset.class_to_idx)
        # y_train = torch.zeros(len(train_dataset), dtype=torch.long)
        #
        # for a in range(len(train_dataset)):
        #     y_train[a] = (train_dataset[a][1])
      
        try:
            N = y_train.shape[0]
        except:
            y_train = np.array(y_train)
            N = y_train.shape[0]
        net_dataidx_map = {i: np.array([], dtype='int64') for i in range(n_nets)}
        assigned_ids = []
        idx_batch = [[] for _ in range(n_nets)]
        num_data_per_client = int(N / n_nets)
        for i in range(n_nets):
            weights = torch.zeros(N)
            proportions = np.random.dirichlet(np.repeat(alpha, K))
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                weights[idx_k] = proportions[k]
            weights[assigned_ids] = 0.0                        #sum(weights)
            try:
                idx_batch[i] = (torch.multinomial(weights, num_data_per_client, replacement=False)).tolist()
            except:
                idx_batch[i] = [i for i in range(N) if i not in assigned_ids]

            assigned_ids += idx_batch[i]

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        client_data_list = [[] for _ in range(self.client_num)]
        client_label_list = [[] for _ in range(self.client_num)]
        client_label_count = {j:{i: 0 for i in range(K)} for j in range(self.client_num)}
        for client_id, client_data, client_label in zip(net_dataidx_map, client_data_list, client_label_list):
            for id in net_dataidx_map[client_id]:
                tr_dat_id = train_dataset[id]
                client_data.append(tr_dat_id)
                client_label_count[client_id][tr_dat_id[1]] += 1
                if tr_dat_id[1] not in client_label:
                    client_label.append(tr_dat_id[1])
        # self.client_label_count = client_label_count

        client_datasets = []
        for client_data in client_data_list:
            # 打乱数据集
            np.random.shuffle(client_data)
            _dataset = ListDataset(client_data)
            client_datasets.append(_dataset)

        return client_datasets, client_label_list, client_label_count

    def _split_dataset_iid(self, dataset, partition_proportions):

        if self.data_set in ['SVHN', 'Tiny-Imagenet']:
            data_labels = dataset.labels
        elif self.data_set in ['FOOD101']:
            data_labels = dataset._labels
        elif self.data_set in ['GTSRB']:
            data_labels = []
            for _, label in dataset._samples:
                data_labels.append(label)
        elif self.data_set in ['CINIC-10']:
            data_labels = dataset.data.targets
            K = len(dataset.data.classes)
        else:
            data_labels = dataset.targets

        # 记录每个K个类别对应的样本下标
        class_idcs = [list(np.argwhere(np.array(data_labels) == y).flatten())
                      for y in range(self.class_num)]

        #记录N个client分别对应样本集合的索引
        client_idcs = [[] for _ in range(self.client_num)]

        for c, fracs in zip(class_idcs, partition_proportions):
            # np.split按照比例将类别为k的样本划分为了N个子集
            # for i, idcs 为遍历第i个client对应样本集合的索引
            np.random.shuffle(c)

            len_ = len(c)
            step = int(len_/self.client_num) + 1

            Evenly_c = [c[i:i+step] for i in range(0, len(c), step)]

            for i, idcs in enumerate(Evenly_c):
                client_idcs[i].extend(idcs)

            # for i, idcs in enumerate(np.split(c, (np.ceil(np.cumsum(fracs)[:-1] * len(c))).astype(int))):
            #     client_idcs[i].extend(list(idcs))
        client_data_list = [[] for _ in range(self.client_num)]
        client_label_list = [[] for _ in range(self.client_num)]
        client_label_count = {j:{i: 0 for i in range(self.class_num)} for j in range(self.client_num)}
        for data_id, client_data, client_label, client_id in zip(client_idcs, client_data_list, client_label_list, client_label_count):
            for id in data_id:
                tr_dat_id = dataset[id]
                client_data.append(tr_dat_id)
                client_label_count[client_id][tr_dat_id[1]] += 1
                if tr_dat_id[1] not in client_label:
                    client_label.append(tr_dat_id[1])
        # self.client_label_count = client_label_count

        # label_partition = []
        # for dtset in client_data_list:
        #     label_partition.append([lb[1] for lb in dtset])
        #
        # save_pkl(label_partition, file_path=f"{os.path.dirname(__file__)}/label_partition/{self.args.data_set}_{self.args.non_iid_alpha}")
        client_datasets = []
        for client_data in client_data_list:
            # 打乱数据集
            np.random.shuffle(client_data)
            _dataset = ListDataset(client_data)
            client_datasets.append(_dataset)

        return client_datasets, client_label_list, client_label_count


if __name__ == "__main__":
    import argparse

    def get_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--batch_size', type=int, default=128)  # 128

        parser.add_argument('--data_set', type=str, default='CIFAR-10')  # BBC-TEXT, AG-NEWS暂时无法使用

        parser.add_argument('--data_partition_mode', type=str, default='non_iid_dirichlet',
                            choices=['iid', 'non_iid_dirichlet'])

        parser.add_argument('--non_iid_alpha', type=float, default=0.5)  # 在进行non_iid_dirichlet数据划分时需要, 该值越大表明数据月均匀

        parser.add_argument('--client_num', type=int, default=10)

        parser.add_argument('--device', type=torch.device, default='cuda')

        parser.add_argument('--seed', type=int, default=0)

        parser.add_argument('--app_name', type=str, default='DecentLaM')

        args = parser.parse_args(args=[])

        return args

    args = get_args()
    dd = DataDistributer(args)
