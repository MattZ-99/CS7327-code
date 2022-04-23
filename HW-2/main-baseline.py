# -*- coding: utf-8 -*-
# @Time : 2022/4/29 13:53
# @Author : Mengtian Zhang
# @Version : v-dev-0.0
# @Function 

"""Summary.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

"""

import torch
from Dataset.DataGenerator import DataGenerator
from train import TrainModuleBaseline
from Network.MLP_for_baseline import MLP
from torch.nn import CrossEntropyLoss
from torch import optim
from tools import utils, funcs

args = funcs.get_args()
utils.seed_everything(args.seed)

data_gen = DataGenerator('./data/data.pkl')
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
trainModule = TrainModuleBaseline(device=device, args=args)

for i in range(len(data_gen)):
    source_dataloader = data_gen.gen_dataloader(item=i, source=True,
                                                batch_size=args.batch_size, drop_last=False, shuffle=True)
    target_dataloader = data_gen.gen_dataloader(item=i, source=False,
                                                batch_size=args.batch_size, drop_last=False, shuffle=True)
    trainModule.set_data_parameters(source_dataloader, target_dataloader)

    net = MLP(in_dim=source_dataloader.dataset.get_data_dim(), label_dim=source_dataloader.dataset.get_label_dim(),
              feature_dim=128, feature_extractor_dims=None, label_classifier_dims=None)
    loss_fn_class = CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.1)
    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    trainModule.set_module_parameters(net=net, loss_fn_class=loss_fn_class, optimizer=optimizer, scheduler=scheduler)

    trainModule.run_train(epochs=args.epochs, tag=f"target-{i}")

trainModule.output_multiple_experiment_results()
trainModule.output_multiple_experiment_results_average()
