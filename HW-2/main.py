# -*- coding: utf-8 -*-
# @Time : 2022/4/23 19:49
# @Author : Mengtian Zhang
# @Version : v-dev-0.0
# @Function 

"""Summary.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

Todo:
    * For module TODOs

"""

import torch
from Dataset.DataGenerator import DataGenerator
from train import TrainModuleDANN
from Network.DANN import DANN
from torch.nn import CrossEntropyLoss
from torch import optim
from tools import utils, funcs

args = funcs.get_args()
utils.seed_everything(args.seed)

data_gen = DataGenerator('./data/data.pkl')
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
trainModule = TrainModuleDANN(device=device, args=args)

for i in range(len(data_gen)):
    source_dataloader = data_gen.gen_dataloader(item=i, source=True,
                                                batch_size=args.batch_size, drop_last=False, shuffle=True)
    target_dataloader = data_gen.gen_dataloader(item=i, source=False,
                                                batch_size=args.batch_size, drop_last=False, shuffle=True)
    trainModule.set_data_parameters(source_dataloader, target_dataloader)

    net = DANN(in_dim=source_dataloader.dataset.get_data_dim(), label_dim=source_dataloader.dataset.get_label_dim(),
               domain_dim=2, feature_dim=128,
               feature_extractor_dims=None, label_classifier_dims=None, domain_classifier_dims=None)
    # print(net)
    loss_fn_class = CrossEntropyLoss()
    loss_fn_domain = CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=0.1)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    trainModule.set_module_parameters(net_dann=net, loss_fn_class=loss_fn_class, loss_fn_domain=loss_fn_domain,
                                      optimizer=optimizer, scheduler=scheduler
                                      )

    trainModule.run_train(epochs=args.epochs, tag=f"target-{i}")

trainModule.output_multiple_experiment_results()
trainModule.output_multiple_experiment_results_average()

