import logging
import math
import os

import numpy as np
import pandas as pd
import torch
import copy
from experiments.models.model import model_pull, generator_model_pull
from lightfed.core import BaseServer
from lightfed.tools.aggregator import ModelStateAvgAgg, NumericAvgAgg
from lightfed.tools.funcs import (consistent_hash, formula, save_pkl, model_size, set_seed, grad_False, grad_True)
from lightfed.tools.model import evaluation, get_buffers, get_parameters, get_cpu_param, DiversityLoss
from torch import nn, optim
from torch.autograd import Variable

from trainer import ClientTrainer
from collections import OrderedDict
import time
import torch.nn.functional as F

class ServerManager(BaseServer):
    def __init__(self, ct, args):
        super().__init__(ct)
        self.super_params = args.__dict__.copy()
        del self.super_params['data_distributer']
        del self.super_params['log_level']
        self.app_name = args.app_name
        self.device = args.device
        self.client_num = args.client_num                   
        self.selected_client_num = args.selected_client_num 
        self.comm_round = args.comm_round                                                        
        self.eval_step_interval = args.eval_step_interval
        # self.eval_on_full_test_data = args.eval_on_full_test_data
        self.temp = args.temp
        self.data_set = args.data_set
        self.weight_agg_plus = args.weight_agg_plus
        self.model_heterogeneity = args.model_heterogeneity
        self.seed = args.seed

        self.model_split_mode = args.model_split_mode
        self.global_model_rate = args.global_model_rate     

        self.full_train_dataloader = args.data_distributer.get_train_dataloader() 
        self.full_test_dataloader = args.data_distributer.get_test_dataloader()    
        self.label_split = args.data_distributer.client_label_list           
        self.client_label_count = args.data_distributer.get_client_label_count()      

        self.local_sample_numbers = [len(args.data_distributer.get_client_train_dataloader(client_id).dataset)
                                     for client_id in range(args.client_num)]
        self.n_class = args.data_distributer.class_num

        self.label_weights, self.label_num, self.qualified_labels = self.get_label_weights()

        self.tran_strategy = args.tran_strategy

        set_seed(args.seed + 657)

        self.model = model_pull(self.super_params).to(self.device)
        
        if self.weight_agg_plus == True:
            self.global_params = get_parameters(self.model.state_dict())
            if self.model_heterogeneity == True:
                self.tmp_counts = {}
                for k, v in self.global_params.items():
                    self.tmp_counts[k] = torch.ones_like(v)
            
            if self.model_heterogeneity == False:  ## model homogeneity
                self.global_params_aggregator = ModelStateAvgAgg()
            
        self.rate = args.rate

        self.selected_clients = None
        self.param_idx = None

        self.weighting = args.weighting

        self.model_idxs = {}
        self.roll_idx = {}

        torch.cuda.empty_cache()

        self.client_collect_list = []

        
        self.client_test_acc_aggregator = NumericAvgAgg()

        self.comm_load = {client_id: 0 for client_id in range(args.client_num)}

        self.client_eval_info = []  
        self.global_train_eval_info = []  

        self.unfinished_client_num = -1

        self.mask_ensemble_gen = args.mask_ensemble_gen
        self.mask_ensemble_glo = args.mask_ensemble_glo

        ## Load generator
        self.generator_1 = generator_model_pull(self.super_params).to(self.device)

        self.save_generator = args.save_generator

        self.batch_size = args.batch_size
        self.eta_g = args.eta_g
        self.I_g = args.I_g
        self.b1 = args.b1
        self.b2 = args.b2
        self.latent_dim = args.latent_dim

        self.eta_d = args.eta_d
        self.I_d = args.I_d

        self.I = args.I

        self.beta_tran = args.beta_tran
        self.beta_div = args.beta_div
        self.beta_ey = args.beta_ey

        self.gamma = args.gamma

        self.noise_label_combine = args.noise_label_combine

        ##
        self.init_loss_fn()

        self.step = -1
    
    def get_label_weights(self):
        label_num = []
        label_weights = []
        qualified_labels = []
        for label in range(self.n_class):
            weights = []
            if len(self.client_label_count) == self.client_num:
                for user in self.client_label_count:
                    weights.append(self.client_label_count[user][label])
                
            else:
                for user in range(self.client_num):
                    if user in self.client_label_count:
                        weights.append(self.client_label_count[user][label])
                    else:
                        weights.append(1)
                        
            if np.max(weights) > 1:
                qualified_labels.append(label)

            # uniform
            label_num.append(np.array(weights))
            label_weights.append( np.array(weights) / np.sum(weights) )
        label_weights = np.array(label_weights).reshape((self.n_class, -1))
        label_num = np.array(label_num).reshape((self.n_class, -1))
        return label_weights, label_num, qualified_labels
    
    def init_loss_fn(self):
        self.KL_batchmean = nn.KLDivLoss(reduction="batchmean").to(self.device)
        self.CE = nn.CrossEntropyLoss().to(self.device)
        self.diversity_loss = DiversityLoss(metric='l2').to(self.device)
        self.KL = nn.KLDivLoss(reduce=False).to(self.device)  # ,log_target=True)
        self.NLL = nn.NLLLoss(reduce=False).to(self.device)
        self.MSE = nn.MSELoss(reduce=False).to(self.device)

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
        if self.step <= self.comm_round - 1:
            if self.weight_agg_plus == False:
                for id_ in range(len(self.selected_clients)):
                    client_id = self.selected_clients[id_]
                    self._ct_.get_node('client', client_id) \
                        .fed_client_train_step(step=self.step, global_params=None) #self.global_params
            else:
                if self.model_heterogeneity == True:
                    local_parameters, self.param_idx = self.distribute() ##
                    for id_ in range(len(self.selected_clients)):
                        client_id = self.selected_clients[id_]
                        self._ct_.get_node('client', client_id) \
                            .fed_client_train_step(step=self.step, global_params=local_parameters[id_]) #self.global_params
                    torch.cuda.empty_cache()
                else:
                    for id_ in range(len(self.selected_clients)):
                        client_id = self.selected_clients[id_]
                        self._ct_.get_node('client', client_id) \
                            .fed_client_train_step(step=self.step, global_params=self.global_params) #self.global_params

    def _new_train_workload_arrage(self):
        if self.selected_client_num < self.client_num:
            selected_client = np.random.choice(range(self.client_num), self.selected_client_num, replace=False)
        elif self.selected_client_num == self.client_num:
            selected_client = np.array([i for i in range(self.client_num)])
        return selected_client
    

    def random_choice_y(self, batch_size, label_num=[]):
        if len(label_num) > 0:
            _label_num = label_num.sum(axis=1)
            label_pop = _label_num / sum(_label_num)
            label_cumpop = np.cumsum(label_pop)
            label_cumpop = np.insert(label_cumpop, 0, 0.0)
            r_bz = np.random.random(batch_size)
            y = []
            for r in r_bz:
                for i in range(len(label_cumpop) - 1):
                    if r >= label_cumpop[i] and r <= label_cumpop[i + 1]:
                        y.append(i)
        else:
            y = np.random.choice([i for i in range(self.n_class)], batch_size)
        return y

    def fed_finish_client_train_step(self,
                                     step,
                                     client_id,
                                     client_model,
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

        if self.weight_agg_plus == True:
            if self.model_heterogeneity == False:
                client_model_params = get_parameters(client_model.state_dict())
                self.global_params_aggregator.put(client_model_params, weight)

        grad_False(client_model)  #
        self.client_collect_list.append((client_id, client_model))

        if self.comm_load[client_id] == 0:
            client_model_params = get_parameters(client_model.state_dict())
            self.comm_load[client_id] = model_size(client_model_params) / 1024 / 1024  ##

        self.unfinished_client_num -= 1
        if not self.unfinished_client_num:
            self.server_train_test_res = {'comm. round': self.step, 'client_id': 'server'}

            if self.model_heterogeneity == True:
                if self.weight_agg_plus == True:
                    self.combine(self.client_collect_list, self.param_idx)
                    self.model.load_state_dict(self.global_params, strict=True)
            else:
                if self.weight_agg_plus == True:
                    self.global_params = self.global_params_aggregator.get_and_clear()
                    self.model.load_state_dict(self.global_params, strict=True)
            torch.cuda.empty_cache()
            
            client_test_acc_avg = self.client_test_acc_aggregator.get_and_clear()
            print('comm. round: {}, client_test_acc: {}'.format(self.step, client_test_acc_avg))

            ##########data-free KD
            self.FloatTensor = torch.cuda.FloatTensor
            self.LongTensor = torch.cuda.LongTensor
            for i in range(self.I):
                # Sample noise as generator input
                self.z1 = Variable(self.FloatTensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))))
                self.y1 = self.random_choice_y(self.batch_size, self.label_num)

                ##update dual generators
                self.Update_generator(self.generator_1, self.z1, self.y1)

                ##update global model
                self.Update_global_model()

                if self.save_generator == True:
                    if i in [0, int(self.I * 0.2)-1, int(self.I * 0.4)-1, int(self.I * 0.6)-1, int(self.I * 0.8)-1, int(self.I * 1.0)-1]:
                        torch.save((self.generator_1), \
                        f"{os.path.dirname(__file__)}/generator/{self.data_set}_{self.noise_label_combine}_{str(i)}_{self.seed}.pth")

                if self.data_set in ['FOOD101', 'Tiny-Imagenet']: # , 'CIFAR-100'
                    if i % 5 == 0:
                        self._set_global_train_eval_info(i)
                        self.global_train_eval_info.append(self.server_train_test_res)
                        self.server_train_test_res = {}
                else:
                    self._set_global_train_eval_info(i)
                    self.global_train_eval_info.append(self.server_train_test_res)
                    self.server_train_test_res = {}

            logging.debug(f"train comm. round:{step} is finished")
            self.client_collect_list = []
            self.next_step()

    def Update_generator(self, trian_generator, update_z, update_y):
        ##update generator_1
        ## need local models, global model, generator_2
        grad_True(trian_generator)
        trian_generator.train()

        grad_False(self.model)
        self.model.eval()
        LOSS_GEN = 0
        L_FID, L_TRAN, L_DIV, L_EY = 0, 0, 0, 0
        ##optimizer
        optimizer_G = optim.Adam(trian_generator.parameters(), lr=self.eta_g, betas=(self.b1, self.b2))
        for _ in range(self.I_g):
            trian_generator.zero_grad(set_to_none=True)
            optimizer_G.zero_grad(set_to_none=True)

            gen_labels = Variable(self.LongTensor(np.array(update_y)))

            # Generate a batch of images
            gen_imgs, h, e_y = trian_generator(update_z, gen_labels)

            if self.noise_label_combine in ['mul', 'add', 'cat']:
                norm_ey = torch.norm(e_y, p=2, dim=1, keepdim=True) - self.gamma * np.sqrt(self.latent_dim) * torch.ones(e_y.shape[0], 1).to(self.device)
                L_ey = torch.mean(torch.max(norm_ey, torch.zeros(e_y.shape[0], 1).to(self.device)))
            
            # feed gen_imgs to global_model
            global_logits = self.model(gen_imgs)
            global_logits_softmax = F.softmax(global_logits / self.temp, dim=1).clone().detach()
            ##pre_res
            _, global_pre = torch.max(global_logits, -1)
            global_pre_res = global_pre.eq(gen_labels)
        
            L_kl_logit = self.cal_client_logits(gen_imgs, update_y)

            L_fid = self.CE(L_kl_logit, gen_labels) # L_cls = torch.mean(self.NLL(F.log_softmax(L_kl_logit, dim=1), gen_labels))

            _, ensemble_pre = torch.max(L_kl_logit, -1)
            ensemble_pre_res = ensemble_pre.eq(gen_labels)

            varepsilon = []
            for idx, pre in enumerate(zip(ensemble_pre_res, global_pre_res)):
                en_pre, gl_pre = pre[0], pre[1]
                if self.tran_strategy == '1':
                    if en_pre == True and gl_pre == False:
                        varepsilon.append(idx)
                    else:
                        continue
                elif self.tran_strategy == '2':
                    varepsilon.append(idx)
                elif self.tran_strategy == '3':
                    if (en_pre == True and gl_pre == False) or (en_pre == False and gl_pre == True):
                        varepsilon.append(idx)
                    else:
                        continue

            clients_logit_log_softmax = F.log_softmax(L_kl_logit / self.temp, dim=1)
            L_tran = - self.KL_batchmean(clients_logit_log_softmax[varepsilon], global_logits_softmax[varepsilon]) ##
                
            L_div = self.diversity_loss(noises=h, layer=gen_imgs)
                        
            if self.noise_label_combine in ['mul', 'add', 'cat']:
                loss_gen =  L_fid + self.beta_tran * L_tran + self.beta_div * L_div # + self.beta_ey * L_ey
            else:
                loss_gen =  L_fid + self.beta_tran * L_tran + self.beta_div * L_div 
            loss_gen.backward()
            optimizer_G.step()

            LOSS_GEN += loss_gen.detach().cpu().numpy()
            L_FID += L_fid.detach().cpu().numpy()
            L_TRAN += L_tran.detach().cpu().numpy()
            L_DIV += L_div.detach().cpu().numpy()
            try:
                L_EY += L_ey.detach().cpu().numpy()
            except:
                L_EY = 0

        # print(info_gen)
        
        info_gen = {'comm. round': self.step, 'client_id': 'server', 'update': 'generator',
                'LOSS_GEN': round(LOSS_GEN / self.I_g, 4), 'L_FID': round(L_FID / self.I_g, 4),
                'L_TRAN': round(L_TRAN / self.I_g, 4), 'L_DIV': round(L_DIV / self.I_g, 4), 'L_EY': round(L_EY / self.I_g, 4)}
        self.server_train_test_res.update(info_gen)
    
    def Update_global_model(self,):
        grad_False(self.generator_1)
        self.generator_1.eval()

        grad_True(self.model)
        self.model.train()

        LOSS_GLO = 0
        # L_KL_1, L_KL_2 = 0.0, 0.0
        ## optimizer
        optimizer_model = optim.SGD(params=self.model.parameters(), lr=self.eta_d, momentum=0.9, weight_decay=0.0001)
        for _ in range(self.I_d):
            self.model.zero_grad(set_to_none=True)
            optimizer_model.zero_grad(set_to_none=True)

            # Sample noise as generator input
            loss_glo = self.cal_L_kl(self.generator_1, self.z1, self.y1)

            loss_glo.backward()
            optimizer_model.step()
            LOSS_GLO += loss_glo.detach().cpu().numpy()
            

        info_global = {'comm. round': self.step, 'client_id': 'server', 'update': 'global_model',
                'LOSS_GLO': round(LOSS_GLO / self.I_d, 4)} #, 'L_KL_1': round(L_KL_1 / self.I_d, 4), 'L_KL_2': round(L_KL_2 / self.I_d, 4)
        # print(info_global)
        self.server_train_test_res.update(global_model_update=info_global)

    
    def cal_L_kl(self, generator, z, y):
        gen_labels = Variable(self.LongTensor(np.array(y)))
        # Generate a batch of images
        gen_imgs, _, _ = generator(z, gen_labels)
        # feed gen_imgs to global_model
        global_logits = self.model(gen_imgs)
        global_logits_log_softmax = F.log_softmax(global_logits / self.temp, dim=1)

        L_kl_logit = self.cal_client_logits(gen_imgs, y)

        clients_logit_softmax = F.softmax(L_kl_logit / self.temp, dim=1).clone().detach()
        L_kl = self.KL_batchmean(global_logits_log_softmax, clients_logit_softmax) 
        return L_kl
    
    def cal_client_logits(self, gen_imgs, y):
        L_kl_logit = 0
        for client_idx, local_model in self.client_collect_list:
            local_model.eval()

            weight = self.label_weights[y][:, client_idx]

            if self.mask_ensemble_gen == True:
                client_logits = local_model(gen_imgs, label_list=self.label_split[client_idx])
            else:
                client_logits = local_model(gen_imgs)
            
            L_kl_logit += client_logits * torch.tensor(weight.reshape(-1, 1), dtype=torch.float32).to(self.device)
        return L_kl_logit
    
    def distribute(self,):
        self.model_rate = np.array(self.rate)
        if self.model_split_mode == 'roll':
            param_idx = self.split_model_roll()
        elif self.model_split_mode == 'static':
            param_idx = self.split_model_static()
        elif self.model_split_mode == 'random':
            param_idx = self.split_model_random()

        self.global_params = get_parameters(self.model.state_dict())
        local_parameters = [OrderedDict() for _ in range(len(self.selected_clients))]
        for k, v in self.global_params.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(self.selected_clients)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            local_parameters[m][k] = copy.deepcopy(v[torch.meshgrid(param_idx[m][k])])
                        else:
                            local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                    else:
                        local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                elif parameter_type in ['running_mean', 'running_var']:
                    local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                else:
                    local_parameters[m][k] = copy.deepcopy(v)
        return local_parameters, param_idx
     
    def split_model_roll(self):
        idx_i = [None for _ in range(len(self.selected_clients))]
        idx = [OrderedDict() for _ in range(len(self.selected_clients))]
        for k, v in self.global_params.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(self.selected_clients)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'conv' in k: # 'conv1' in k or 'conv2' in k or 'blocks' in k
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[self.selected_clients[m]] / self.global_model_rate
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                
                                roll = self.step % output_size
                                model_idx = torch.arange(output_size, device=v.device)
                                model_idx = torch.roll(model_idx, roll, -1)
                                output_idx_i_m = model_idx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            elif 'shortcut' in k:
                                input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                output_idx_i_m = idx_i[m]
                            elif 'linear' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            else:
                                raise ValueError('Not valid k')
                            idx[m][k] = (output_idx_i_m, input_idx_i_m) ##
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'linear' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                elif parameter_type in ['running_mean', 'running_var']:
                    input_idx_i_m = idx_i[m]
                    idx[m][k] = input_idx_i_m

                else:
                    pass
        return idx
    
    def split_model_static(self):
        idx_i = [None for _ in range(len(self.selected_clients))]
        idx = [OrderedDict() for _ in range(len(self.selected_clients))]
        for k, v in self.global_params.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(self.selected_clients)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'conv' in k: # 'conv1' in k or 'conv2' in k or 'blocks' in k
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[self.selected_clients[m]] / self.global_model_rate
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                model_idx = torch.arange(output_size, device=v.device)
                                output_idx_i_m = model_idx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            elif 'shortcut' in k:
                                input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                output_idx_i_m = idx_i[m]
                            elif 'linear' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            else:
                                raise ValueError('Not valid k')
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'linear' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                elif parameter_type in ['running_mean', 'running_var']:
                    input_idx_i_m = idx_i[m]
                    idx[m][k] = input_idx_i_m

                else:
                    pass
        return idx
    
    def split_model_random(self):
        idx_i = [None for _ in range(len(self.selected_clients))]
        idx = [OrderedDict() for _ in range(len(self.selected_clients))]
        for k, v in self.global_params.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(self.selected_clients)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'conv' in k: # 'conv1' in k or 'conv2' in k or 'blocks' in k
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[self.selected_clients[m]] / self.global_model_rate
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                model_idx = torch.randperm(output_size, device=v.device)
                                output_idx_i_m = model_idx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            elif 'shortcut' in k:
                                input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                output_idx_i_m = idx_i[m]
                            elif 'linear' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            else:
                                raise ValueError('Not valid k')
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'linear' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m

                elif parameter_type in ['running_mean', 'running_var']:
                    input_idx_i_m = idx_i[m]
                    idx[m][k] = input_idx_i_m

                else:
                    pass
        return idx

    def combine(self, local_models, param_idx): #, user_idx
        count = OrderedDict()
        updated_parameters = copy.deepcopy(self.global_params) ## 
        tmp_counts_cpy = copy.deepcopy(self.tmp_counts)        ## 
        for k, v in updated_parameters.items():
            parameter_type = k.split('.')[-1]
            count[k] = v.new_zeros(v.size(), dtype=torch.float32, device=self.device)
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32, device=self.device)
            for m, local_m in enumerate(local_models):
                local_pars = get_parameters(local_m[1].state_dict())
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            if 'linear' in k:
                                label_split = self.label_split[local_m[0]]  ##
                                param_idx[m][k] = list(param_idx[m][k])
                                param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                tmp_v[torch.meshgrid(param_idx[m][k])] += self.tmp_counts[k][torch.meshgrid(param_idx[m][k])] * local_pars[k][label_split]
                                count[k][torch.meshgrid(param_idx[m][k])] += self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]
                                tmp_counts_cpy[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                output_size = v.size(0)
                                scaler_rate = self.model_rate[local_m[0]] / self.global_model_rate
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                if self.weighting == 'avg':
                                    K = 1
                                elif self.weighting == 'width':
                                    K = local_output_size
                                elif self.weighting == 'updates':
                                    K = self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]
                                elif self.weighting == 'updates_width':
                                    K = local_output_size * self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]

                                tmp_v[torch.meshgrid(param_idx[m][k])] += K * local_pars[k]
                                count[k][torch.meshgrid(param_idx[m][k])] += K
                                tmp_counts_cpy[k][torch.meshgrid(param_idx[m][k])] += 1
                        else:
                            tmp_v[param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]] * local_pars[k]
                            count[k][param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]]
                            tmp_counts_cpy[k][param_idx[m][k]] += 1
                    else:
                        if 'linear' in k:
                            label_split = self.label_split[local_m[0]]
                            param_idx[m][k] = param_idx[m][k][label_split]
                            tmp_v[param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]] * local_pars[k][label_split]
                            count[k][param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]]
                            tmp_counts_cpy[k][param_idx[m][k]] += 1
                        else:
                            tmp_v[param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]] * local_pars[k]
                            count[k][param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]]
                            tmp_counts_cpy[k][param_idx[m][k]] += 1
                elif parameter_type in ['running_mean', 'running_var']:
                    tmp_v[param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]] * local_pars[k]
                    count[k][param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]]
                    tmp_counts_cpy[k][param_idx[m][k]] += 1
                else:
                    tmp_v += self.tmp_counts[k] * local_pars[k]
                    count[k] += self.tmp_counts[k]
                    tmp_counts_cpy[k] += 1
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
            self.tmp_counts = copy.deepcopy(tmp_counts_cpy)
        self.global_params = updated_parameters
        return
        
    def _set_global_train_eval_info(self, i):
        # loss, acc, num = evaluation(model=self.model,
        #                             dataloader=self.full_train_dataloader,
        #                             criterion=self.CE,
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
        self.server_train_test_res.update(iter=i, test_loss=loss, test_acc=acc, test_sample_size=num)
        test_ = {}
        test_.update(iter=i, test_loss=loss, test_acc=acc, test_sample_size=num)
        print(test_)

        # logging.info(f"global eval info:{self.server_train_test_res}")

class ClientManager(BaseServer):
    def __init__(self, ct, args):
        super().__init__(ct)
        self.I_c = args.I_c
        self.device = args.device
        self.client_id = self._ct_.role_index
        self.model_type = args.model_type
        self.data_set = args.data_set
        self.weight_agg_plus = args.weight_agg_plus
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
        if not os.path.exists(f"{self.save_local_model_dir}/{self.data_set}_{self.args['model_heterogeneity']}_{self.args['model_here_level']}_{self.client_id}_{self.model_type}_{self.client_num}_{step}_{self.I_c}_{self.eta_c}_{self.args['non_iid_alpha']}_{self.args['seed']}.pth"):
            logging.debug(f"training client_id:{self.client_id}, comm. round:{step}")

            self.trainer.res = {'communication round': step, 'client_id': self.client_id}
            # self.trainer.res = {'client_id': self.client_id}
            
            if self.weight_agg_plus == False:
                if step == 0:
                    self.trainer.pull_local_model(self.args, model_rate = self.args['rate'][self.client_id])
            else:
                self.trainer.pull_local_model(self.args, model_rate = self.args['rate'][self.client_id])
                self.trainer.model.load_state_dict(global_params, strict=True)

            self.trainer.train_locally_step(self.I_c)
            
            torch.save((self.trainer.model, self.trainer.res), \
                        f"{self.save_local_model_dir}/{self.data_set}_{self.args['model_heterogeneity']}_{self.args['model_here_level']}_{self.client_id}_{self.model_type}_{self.client_num}_{step}_{self.I_c}_{self.eta_c}_{self.args['non_iid_alpha']}_{self.args['seed']}.pth")

            self.finish_train_step(self.trainer.model, self.trainer.res)
            self.trainer.clear()
            torch.cuda.empty_cache()
        else:
            if self.args['model_heterogeneity'] == False:
                try:
                    trained_local_model, local_res, _ = torch.load(f"{self.save_local_model_dir}/{self.data_set}_{self.client_id}_{self.model_type}_{self.client_num}_{step}_{self.I_c}_{self.eta_c}_{self.args['non_iid_alpha']}_{self.args['seed']}.pth")
                except:
                    trained_local_model, local_res = torch.load(f"{self.save_local_model_dir}/{self.data_set}_{self.client_id}_{self.model_type}_{self.client_num}_{step}_{self.I_c}_{self.eta_c}_{self.args['non_iid_alpha']}_{self.args['seed']}.pth")
            else:
                try:
                    trained_local_model, local_res, _ = torch.load(f"{self.save_local_model_dir}/{self.data_set}_{self.args['model_heterogeneity']}_{self.args['model_here_level']}_{self.client_id}_{self.model_type}_{self.client_num}_{step}_{self.I_c}_{self.eta_c}_{self.args['non_iid_alpha']}_{self.args['seed']}.pth")
                except:
                    trained_local_model, local_res = torch.load(f"{self.save_local_model_dir}/{self.data_set}_{self.args['model_heterogeneity']}_{self.args['model_here_level']}_{self.client_id}_{self.model_type}_{self.client_num}_{step}_{self.I_c}_{self.eta_c}_{self.args['non_iid_alpha']}_{self.args['seed']}.pth")

            self.finish_train_step(trained_local_model, local_res)

    def finish_train_step(self, model, res):
        logging.debug(f"finish_train_step comm. round:{self.step}, client_id:{self.client_id}")

        self._ct_.get_node("server") \
            .set(deepcopy=False) \
            .fed_finish_client_train_step(self.step,
                                          self.client_id,
                                          model,
                                          res)