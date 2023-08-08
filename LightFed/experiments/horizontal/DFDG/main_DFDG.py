from lightfed.core import Config

from manager import ClientManager, ServerManager
from param_config import get_args

import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# torch.cuda.device_count()
# torch.cuda.get_device_name()
# torch.cuda.current_device()


if __name__ == "__main__":
    args = get_args()
    
    Config() \
        .add_role("server", 1, lambda ct: ServerManager(ct, args)) \
        .add_role("client", args.client_num, lambda ct: ClientManager(ct, args)) \
        .init_log(args.log_level) \
        .set_proc_title(args.app_name) \
        .get_runner() \
        .run()

"""
cd /root/luoky/LightFed/experiments/horizontal/fedavg
export PYTHONPATH=/root/luoky/LightFed
mpiexec -n 6 python main_scaffold.py
"""
