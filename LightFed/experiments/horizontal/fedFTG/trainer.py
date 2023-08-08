import logging
from collections import OrderedDict

import torch
from experiments.models.model import model_pull
from lightfed.tools.funcs import set_seed, grad_True
from lightfed.tools.model import evaluation, CycleDataloader, get_parameters
from torch import nn
from collections import Counter

class ClientTrainer:
    def __init__(self, args, client_id):
        self.client_id = client_id
        self.device = args.device
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay
        self.eta_c = args.eta_c
        self.data_set = args.data_set

        self.train_dataloader = args.data_distributer.get_client_train_dataloader(client_id)
        # self.train_batch_data_iter = CycleDataloader(self.train_dataloader)
        self.train_label_list = args.data_distributer.get_client_label_list(client_id)
        if self.data_set == 'CINIC-10':
            self.test_dataloader = args.data_distributer.get_client_test_dataloader(client_id)
        else:
            self.test_dataloader = args.data_distributer.get_test_dataloader()

        self.unique_labels = args.data_distributer.class_num

        self.res = {}

        set_seed(args.seed + 657)
        
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        # self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.eta_c, betas=(0.9, 0.999), eps=1e-08,
                                          # weight_decay=1e-2, amsgrad=False)
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)

    def pull_local_model(self, args, model_rate):
        self.model = model_pull(args, model_rate=model_rate).to(self.device)

    def clear(self):
        self.res = {}
        # self.model = None
        self.optimizer = None
        torch.cuda.empty_cache()

    def train_locally_step(self, I):
        self.res.update(epoch=[], test_loss=[], test_acc=[], test_sample_size=[], m_LOSS=[])
        grad_True(self.model)
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.eta_c, weight_decay=self.weight_decay)
        self.model.train()
        LOSS = 0
        for epoch in range(I):
            self.model.zero_grad(set_to_none=True)
            self.optimizer.zero_grad(set_to_none=True)
            # lll = self.model.state_dict()
            ##batch数据
            for x, y in self.train_dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                logit = self.model(x) # , label_list=self.train_label_list
                loss = self.criterion(logit, y)
                loss.backward()
                self.optimizer.step()
                LOSS += loss
            
            loss, acc, num = self.get_eval_info(epoch)
            self.res['epoch'].append(epoch)
            self.res['test_loss'].append(loss)
            self.res['test_acc'].append(acc)
            self.res['test_sample_size'].append(num)

            LOSS = LOSS.detach().cpu().numpy() / len(self.train_dataloader)
            self.res['m_LOSS'].append(LOSS)
        # logging.debug(f"train_locally_step for step: {tau}")
        self.model.zero_grad(set_to_none=True)


    def get_eval_info(self, step):

        # loss, acc, num = evaluation(model=self.model,
        #                             dataloader=self.train_dataloader,
        #                             criterion=self.criterion,
        #                             model_params=self.model_params,
        #                             device=self.device)
        # res.update(train_loss=loss, train_acc=acc, train_sample_size=num)

        if self.data_set in ['Tiny-Imagenet', 'FOOD101']: # , 'CIFAR-100'
            if step % 5 == 0:
                loss, acc, num = evaluation(model=self.model,
                                            dataloader=self.test_dataloader,
                                            criterion=self.criterion,
                                            model_params=None,
                                            device=self.device)
                return loss, acc, num
        else:
            loss, acc, num = evaluation(model=self.model,
                                            dataloader=self.test_dataloader,
                                            criterion=self.criterion,
                                            model_params=None,
                                            device=self.device)
            return loss, acc, num
        
