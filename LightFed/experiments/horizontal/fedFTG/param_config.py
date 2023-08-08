import argparse
import logging
import os

import numpy as np
import torch
from experiments.datasets.data_distributer import DataDistributer
from lightfed.tools.funcs import consistent_hash, set_seed

MODEL_SPLIT_RATE = {'a': 1.0, 'b': 0.5, 'c': 0.25, 'd': 0.125, 'e': 0.0625}
RATE_NAME = ['a', 'b', 'c', 'd', 'e']

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--comm_round', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=64)  # 128 

    parser.add_argument('--eval_step_interval', type=int, default=5)

    parser.add_argument('--eval_batch_size', type=int, default=256)

    parser.add_argument('--I_c', type=int, default=50, help='iteration of the training procedure in each client')

    parser.add_argument('--eta_c', type=float, default=0.01, help='learning rate of local models!!!') 

    parser.add_argument('--weight_agg_plus', type=lambda s: s == 'True', default=False)

    parser.add_argument('--model_heterogeneity', type=lambda s: s == 'True', default=True)

    parser.add_argument('--model_here_level', type=int, default=4, choices=[2, 3, 4])

    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--model_type', type=str, default='Lenet', choices=['Lenet', 'ResNet_18', 'ResNet_20', 'ResNet_34', 'ResNet_50'])

    parser.add_argument('--model_norm', type=str, default='bn', choices=['none', 'bn', 'in', 'ln', 'gn'],help='对输出特征的标准化方法，bn表示batch normal')

    parser.add_argument('--model_split_mode', type=str, default='roll', choices=['roll', 'static', 'random', 'None'], help='分解模型的方法')

    parser.add_argument('--global_model_rate', type=float, default=MODEL_SPLIT_RATE['a'], help='全局模型的大小')

    parser.add_argument('--weighting', default='updates', type=str, choices=['avg', 'width', 'updates', 'updates_width'], help='全局模型聚合的方式')

    parser.add_argument('--scale', type=lambda s: s == 'True', default=True)

    parser.add_argument('--mask', type=lambda s: s == 'True', default=False)

    # parser.add_argument('--mask_ensemble_gen', type=bool, default=True)

    # parser.add_argument('--mask_ensemble_glo', type=bool, default=False)

    ###
    parser.add_argument("--I", type=int, default=200, help="training number of adversarial training")

    ###parameters for generator
    parser.add_argument('--generator_model_type', type=str, default='ACGAN')

    # parser.add_argument('--reload_generator', type=lambda s: s == 'True', default=False)

    parser.add_argument("--I_g", type=int, default=20, help="training number of generator")

    parser.add_argument('--eta_g', type=float, default=0.0002, help="adam: learning rate")

    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")

    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")

    parser.add_argument("--beta_md", type=float, default=1.0, help="hyper-parameter of L_md loss")

    parser.add_argument("--beta_div", type=float, default=1.0, help="hyper-parameter of L_div loss")

    # parser.add_argument("--beta_ey", type=float, default=1.0, help="hyper-parameter for ey loss")

    # parser.add_argument("--gamma", type=float, default=1.0, help="hyper-parameter for Embedding")

    # parser.add_argument("--bn_track", type=bool, default=False) ## 确定为False

    parser.add_argument('--noise_label_combine', type=str, default='cat', choices=['mul', 'add', 'cat', 'cat_naive', 'none'])

    parser.add_argument('--save_generator', type=lambda s: s == 'True', default=True)

    ##parameters for global model
    parser.add_argument('--eta_d', type=float, default=0.01, help="SGD: learning rate")

    parser.add_argument("--I_d", type=int, default=2, help="training number of global model")

    parser.add_argument("--temp", type=int, default=10, help="Distillation temperature")

    parser.add_argument('--data_set', type=str, default='CINIC-10',
                        choices=['MNIST', 'FMNIST', 'CIFAR-10', 'CINIC-10', 'CIFAR-100', 'SVHN', 'FOOD101', 'Tiny-Imagenet', 'GTSRB'])

    parser.add_argument('--data_partition_mode', type=str, default='non_iid_unbalanced',
                        choices=['iid', 'non_iid_unbalanced', 'non_iid_balanced']) #'non_iid_dirichlet',

    parser.add_argument('--non_iid_alpha', type=float, default=1.0)  # 在进行non_iid_dirichlet数据划分时需要, 该值越大表明数据越均匀

    parser.add_argument('--client_num', type=int, default=10)

    parser.add_argument('--selected_client_num', type=int, default=10)

    parser.add_argument('--device', type=torch.device, default='cuda')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--log_level', type=logging.getLevelName, default='INFO')

    parser.add_argument('--app_name', type=str, default='FedFTG')

    # args = parser.parse_args(args=[])
    args = parser.parse_args()

    args.rate = _get_model_split_rate(args)
    print(args.rate)

    super_params = args.__dict__.copy()
    del super_params['log_level']
    super_params['device'] = super_params['device'].type
    ff = f"{args.app_name}-{consistent_hash(super_params, code_len=64)}.pkl"
    ff = f"{os.path.dirname(__file__)}/Result/{ff}"
    if os.path.exists(ff):
        print(f"output file existed, skip task")
        exit(0)

    args.data_distributer = _get_data_distributer(args)

    # args.weight_matrix = _get_weight_matrix(args)

    return args

def _get_data_distributer(args):
    set_seed(args.seed + 5363)
    # set_seed(args.seed + 5364)
    return DataDistributer(args)

def _get_model_split_rate(args):
    if args.model_heterogeneity == False:
        sample_rate = [1.0 for _ in range(args.client_num)]
    else:
        sample_rate = []
        for i in range(args.client_num):
            sample_rate.append(cal_level(args.model_here_level, i+1, args.client_num))
    return sample_rate


def cal_level(rho, i, N):
    le = rho * i / N
    return 0.5 ** min([2, np.floor(le)])

# def _get_model_split_rate(args):
#     if args.model_heterogeneity == False:
#         sample_rate = torch.tensor([0 for _ in range(args.client_num)])
#     else:
#         sample_rate = torch.multinomial(torch.tensor([0.1, 0.2, 0.4, 0.6, 0.7]), args.client_num, replacement=True)
#     return [MODEL_SPLIT_RATE[RATE_NAME[sample_rate[i]]] for i in range(args.client_num)]


# def _get_weight_matrix(args):
#     n = args.client_num
#     wm = np.zeros(shape=(n, n))
#     for i in range(n):
#         wm[i][(i - 1 + n) % n] = 1 / 3
#         wm[i][i] = 1 / 3
#         wm[i][(i + 1 + n) % n] = 1 / 3
#
#     assert np.allclose(wm, wm.T)
#     return wm
