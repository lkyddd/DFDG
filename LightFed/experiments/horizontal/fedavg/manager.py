import logging
import math
import os

import numpy as np
import pandas as pd
import torch
from experiments.models.model import model_pull
from lightfed.core import BaseServer
from lightfed.tools.aggregator import ModelStateAvgAgg, NumericAvgAgg
from lightfed.tools.funcs import (consistent_hash, formula, save_pkl, model_size, set_seed)
from lightfed.tools.model import evaluation, get_buffers, get_parameters, get_cpu_param
from torch import nn

from trainer import ClientTrainer
from collections import OrderedDict
import time

class ServerManager(BaseServer):
    def __init__(self, ct, args):
        super().__init__(ct)
        self.super_params = args.__dict__.copy()
        del self.super_params['data_distributer']
        del self.super_params['log_level']
        self.app_name = args.app_name
        self.device = args.device
        self.client_num = args.client_num                   # 
        self.selected_client_num = args.selected_client_num # 
        self.comm_round = args.comm_round                   #                             
        self.eval_step_interval = args.eval_step_interval
        # self.eval_on_full_test_data = args.eval_on_full_test_data
        self.data_set = args.data_set

        self.full_train_dataloader = args.data_distributer.get_train_dataloader()  ##
        self.full_test_dataloader = args.data_distributer.get_test_dataloader()    ##
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        set_seed(args.seed + 657)

        self.model = model_pull(self.super_params).to(self.device)  # 
        # path = os.path.join(os.getcwd() + "/LightFed/experiments/horizontal")
        # # path = os.path.abspath(os.path.join(os.getcwd(), ".."))
        # if not os.path.exists(f"{path}/model_save/{args.model_type}_{args.data_set}.pth"):
        #     torch.save(self.model, f"{path}/model_save/{args.model_type}_{args.data_set}.pth")

        self.global_params = get_cpu_param(self.model.state_dict())
        torch.cuda.empty_cache()

        self.local_sample_numbers = [len(args.data_distributer.get_client_train_dataloader(client_id).dataset)
                                     for client_id in range(args.client_num)]

        self.global_params_aggregator = ModelStateAvgAgg()

        self.client_test_acc_aggregator = NumericAvgAgg()

        self.comm_load = {client_id: 0 for client_id in range(args.client_num)}

        self.client_eval_info = []  #
        self.global_train_eval_info = []  ##

        self.unfinished_client_num = -1

        self.step = -1

    def start(self):
        logging.info("start...")
        self.next_step()

    def end(self):
        logging.info("end...")

        self.super_params['device'] = self.super_params['device'].type

        ff = f"{self.app_name}-{consistent_hash(self.super_params, code_len=64)}.pkl"
        logging.info(f"output to {ff}")

        result = {'super_params': self.super_params,
                  'global_train_eval_info': pd.DataFrame(self.global_train_eval_info),
                  'client_eval_info': pd.DataFrame(self.client_eval_info),
                  'comm_load': self.comm_load}
        save_pkl(result, f"{os.path.dirname(__file__)}/Result/{ff}")

        self._ct_.shutdown_cluster()

    def end_condition(self):
        return self.step > self.comm_round - 1

    def next_step(self):
        self.step += 1
        self.selected_clients = self._new_train_workload_arrage()  ##
        self.unfinished_client_num = self.selected_client_num
        self.global_params_aggregator.clear()
        
        if self.step <= self.comm_round - 1:
            for client_id in self.selected_clients:
                self._ct_.get_node('client', client_id) \
                    .fed_client_train_step(step=self.step, global_params=self.global_params)

    def _new_train_workload_arrage(self):
        if self.selected_client_num < self.client_num:
            selected_client = np.random.choice(range(self.client_num), self.selected_client_num, replace=False)
        elif self.selected_client_num == self.client_num:
            selected_client = np.array([i for i in range(self.client_num)])
        return selected_client


    def fed_finish_client_train_step(self,
                                     step,
                                     client_id,
                                     client_model_params,
                                     eval_info):
        logging.debug(f"train comm. round of client_id:{client_id} comm. round:{step} was finished")
        assert self.step == step
        self.client_eval_info.append(eval_info)

        weight = self.local_sample_numbers[client_id]
        if self.data_set in ['FOOD101', 'Tiny-Imagenet']: # , 'CIFAR-100'
            if self.step % 5 == 0:
                try:
                    self.client_test_acc_aggregator.put(eval_info['test_acc'][-1], weight)
                except:
                    self.client_test_acc_aggregator.put(1.0, weight)
        else:
            try:
                self.client_test_acc_aggregator.put(eval_info['test_acc'][-1], weight)
            except:
                self.client_test_acc_aggregator.put(1.0, weight)


        self.global_params_aggregator.put(client_model_params, weight)

        if self.comm_load[client_id] == 0:
            self.comm_load[client_id] = model_size(client_model_params) / 1024 / 1024  ##

        self.unfinished_client_num -= 1
        if not self.unfinished_client_num:
            self.server_train_test_res = {'comm. round': self.step, 'client_id': 'server'}
            self.global_params = self.global_params_aggregator.get_and_clear()
            client_test_acc_avg = self.client_test_acc_aggregator.get_and_clear()

            print('comm. round: {}, client_test_acc: {}'.format(self.step, client_test_acc_avg))
            self.model.load_state_dict(self.global_params, strict=True)

            self._set_global_train_eval_info()

            logging.debug(f"train comm. round:{step} is finished")
            self.global_train_eval_info.append(self.server_train_test_res)
            self.server_train_test_res = {}
            self.next_step()


    def _set_global_train_eval_info(self):
        # loss, acc, num = evaluation(model=self.model,
        #                             dataloader=self.full_train_dataloader,
        #                             criterion=self.criterion,
        #                             model_params=self.global_params,
        #                             device=self.device,
        #                             eval_full_data=False)
        # eval_info.update(train_loss=loss, train_acc=acc, train_sample_size=num)
        if self.data_set in ['FOOD101', 'Tiny-Imagenet', 'CINIC-10']:
            loss, acc, num = evaluation(model=self.model,
                                    dataloader=self.full_test_dataloader,
                                    criterion=self.CE,
                                    device=self.device,
                                    eval_full_data=False)
        else:
            loss, acc, num = evaluation(model=self.model,
                                        dataloader=self.full_test_dataloader,
                                        criterion=self.CE,
                                        device=self.device,
                                        eval_full_data=True)
        torch.cuda.empty_cache()
        self.server_train_test_res.update(test_loss=loss, test_acc=acc, test_sample_size=num)

        logging.info(f"global eval info:{self.server_train_test_res}")

class ClientManager(BaseServer):
    def __init__(self, ct, args):
        super().__init__(ct)
        self.I_c = args.I_c
        self.device = args.device
        self.client_id = self._ct_.role_index
        self.model_type = args.model_type
        self.data_set = args.data_set
        self.client_num = args.client_num
        self.eta_c = args.eta_c

        self.args = args.__dict__.copy()
        del self.args['data_distributer']

        self.trainer = ClientTrainer(args, self.client_id)
        self.step = 0

        self.save_local_model_dir = os.path.abspath(os.path.join(__file__, "../../trained_local_models"))


    def start(self):
        logging.info("start...")

    def end(self):
        logging.info("end...")

    def end_condition(self):
        return False

    def fed_client_train_step(self, step, global_params):
        self.step = step
        if not os.path.exists(f"{self.save_local_model_dir}/{self.data_set}_{self.client_id}_{self.model_type}_{self.client_num}_{step}_{self.I_c}_{self.eta_c}_{self.args['non_iid_alpha']}_{self.args['seed']}.pth"):
            logging.debug(f"training client_id:{self.client_id}, comm. round:{step}")

            self.trainer.res = {'communication round': step, 'client_id': self.client_id}

            self.trainer.pull_local_model(self.args)  ##
            # path = os.path.abspath(os.path.join(os.getcwd(), ".."))
            # self.trainer.model = torch.load(f"{path}/model_save/{self.model_type}_{self.data_set}.pth")
            self.trainer.model.load_state_dict(global_params, strict=True)

            self.trainer.train_locally_step(self.I_c)
            torch.save((self.trainer.model, self.trainer.res), \
                        f"{self.save_local_model_dir}/{self.data_set}_{self.client_id}_{self.model_type}_{self.client_num}_{step}_{self.I_c}_{self.eta_c}_{self.args['non_iid_alpha']}_{self.args['seed']}.pth")

            model_params = get_cpu_param(self.trainer.model.state_dict())
            self.finish_train_step(model_params, self.trainer.res)
            self.trainer.clear()
            torch.cuda.empty_cache()
        else:
            try:
                trained_local_model, local_res, _ = torch.load(f"{self.save_local_model_dir}/{self.data_set}_{self.client_id}_{self.model_type}_{self.client_num}_{step}_{self.I_c}_{self.eta_c}_{self.args['non_iid_alpha']}_{self.args['seed']}.pth")
            except:
                trained_local_model, local_res = torch.load(f"{self.save_local_model_dir}/{self.data_set}_{self.client_id}_{self.model_type}_{self.client_num}_{step}_{self.I_c}_{self.eta_c}_{self.args['non_iid_alpha']}_{self.args['seed']}.pth")

            self.finish_train_step(trained_local_model.state_dict(), local_res)

    def finish_train_step(self, model_params, res):
        logging.debug(f"finish_train_step comm. round:{self.step}, client_id:{self.client_id}")

        self._ct_.get_node("server") \
            .set(deepcopy=False) \
            .fed_finish_client_train_step(self.step,
                                          self.client_id,
                                          model_params,
                                          res)