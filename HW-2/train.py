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
from tqdm import tqdm
import math
import os
from tools import utils


class TrainModule:
    def __init__(self, device, args):
        self.target_dataloader = None
        self.source_dataloader = None
        self.net = None
        self.scheduler = None
        self.optimizer = None

        self.device = device
        self.args = args

        self.output_dir = args.output_dir

        self._init_log_parameters_global()

    def _init_output_dir(self, name: str):
        self.output_dir = os.path.join(self.args.output_dir, name)
        self.output_dir = os.path.join(self.output_dir, utils.get_timestamp())
        utils.makedirs(self.output_dir)

    def set_data_parameters(self, source_dataloader, target_dataloader):
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader

    def set_module_parameters(self, *args, **kwargs):
        raise NotImplementedError

    def _init_log_parameters(self):
        self.log_train_dict = {
            'train_max_sr_class_acc': 0,
            'train_max_sr_class_acc_epoch': -1,
            'result_max': {},
        }

    def log_train_step(self, result: dict, epoch):
        # condition_log_para_update = result['sr_acc_class'] > self.log_train_dict['train_max_sr_class_acc']
        condition_log_para_update = True
        if condition_log_para_update:
            self.log_train_dict['train_max_sr_class_acc'] = result['sr_acc_class']
            self.log_train_dict['train_max_sr_class_acc_epoch'] = epoch
            self.log_train_dict['result_max'] = result

    def log_train_final_result(self, tag, result, epoch):
        self.train_final_dict[tag] = {
            "result": result,
            "epoch": epoch
        }

    def _init_log_parameters_global(self):
        self.train_final_dict = {}

    def output_final_max_sr_class_acc(self, tag, result, epoch_max=-1):
        eval_result = result
        output = ""
        for k in eval_result:
            output += f"  * {k}={eval_result[k]}, \n"
        output_star = '*' * 100
        output = f"\n{output_star}\n" \
                 + f"[{tag}]\tmaximum source class accuracy epoch: {epoch_max}\n" \
                 + f"{output}\n" \
                 + f"{output_star}\n"
        print(output)

        self._save_output_file(output, os.path.join(self.output_dir, "results.txt"))

    @staticmethod
    def _save_output_file(output, file):
        with open(file, 'a') as f:
            f.write(output)
            f.write('\n')

    def output_multiple_experiment_results(self):
        for tag in self.train_final_dict:
            result = self.train_final_dict[tag]["result"]
            epoch = self.train_final_dict[tag]["epoch"]
            self.output_final_max_sr_class_acc(tag, result, epoch)

    def output_multiple_experiment_results_average(self):
        result_list = list()
        for tag in self.train_final_dict:
            result = self.train_final_dict[tag]["result"]
            result_list.append(result)
        result_keys = list(result_list[0].keys())
        average_result_dict = dict()
        for key in result_keys:
            _sum = 0
            for r in result_list:
                _sum += r[key]
            average_result_dict[key] = _sum / len(result_list)
        self.output_final_max_sr_class_acc("Average", average_result_dict)

    def train_one_step(self, epoch, epochs):
        raise NotImplementedError

    def eval_one_step(self):
        raise NotImplementedError

    def run_train(self, epochs=10, tag=None):
        self._init_log_parameters()
        run_bar = tqdm(range(epochs + 1))
        for epoch in run_bar:
            run_bar.set_description(f"[{tag}] [Epoch={epoch}/{epochs}]")
            self.net.train()
            self.train_one_step(epoch, epochs)

            self.net.eval()
            eval_result = self.eval_one_step()
            run_bar.set_postfix({
                "sr_acc_class": eval_result['sr_acc_class'],
                'tg_acc_class': eval_result['tg_acc_class'],
            })

            self.scheduler.step()

            self.log_train_step(eval_result, epoch)

        # self.output_final_max_sr_class_acc(tag, self.log_train_dict['result_max'],
        #                                    self.log_train_dict['train_max_sr_class_acc_epoch'])
        self.log_train_final_result(tag, self.log_train_dict['result_max'],
                                    self.log_train_dict['train_max_sr_class_acc_epoch'])


class TrainModuleDANN(TrainModule):
    def __init__(self, device, args):
        super(TrainModuleDANN, self).__init__(device, args)

        self.loss_fn_class = None
        self.loss_fn_domain = None

        self._init_output_dir("dann")

    def set_module_parameters(self, net_dann, loss_fn_class, loss_fn_domain, optimizer, scheduler):
        self.net = net_dann.to(self.device)
        self.loss_fn_class = loss_fn_class
        self.loss_fn_domain = loss_fn_domain
        self.optimizer = optimizer
        self.scheduler = scheduler

    @staticmethod
    def _get_alpha(p: float = 0, gamma: float = 10.):
        return 2 / (1 + math.exp(-gamma * p)) - 1

    def train_one_step(self, epoch: int = -1, total_epochs: int = 10):
        source_dataloader = self.source_dataloader
        target_dataloader = self.target_dataloader
        net = self.net
        device = self.device

        steps_dataloader = min(len(source_dataloader), len(target_dataloader))
        iter_source = iter(source_dataloader)
        iter_target = iter(target_dataloader)

        alpha = self.args.alpha_cft * self._get_alpha(p=(1. * epoch / total_epochs))

        for i in range(steps_dataloader):
            net.set_source()
            sr_data, sr_label = next(iter_source)
            sr_domain = torch.zeros_like(sr_label)
            sr_data, sr_label, sr_domain = sr_data.to(device), sr_label.to(device), sr_domain.to(device)
            sr_out_label, sr_out_domain = net(sr_data, alpha=alpha)

            loss_sr_class = self.loss_fn_class(sr_out_label, sr_label)
            loss_sr_domain = self.loss_fn_domain(sr_out_domain, sr_domain)

            net.set_target()
            tg_data, tg_label = next(iter_target)
            tg_domain = torch.ones_like(tg_label)
            tg_data, tg_domain = tg_data.to(device), tg_domain.to(device)
            tg_out_domain = net(tg_data)

            loss_tg_domain = self.loss_fn_domain(tg_out_domain, tg_domain)

            self.optimizer.zero_grad()
            loss = loss_sr_class + loss_sr_domain + loss_tg_domain
            # loss = loss_sr_class
            loss.backward()
            self.optimizer.step()

    def eval_one_dataloader(self, dataloader, domain_flag=0):
        device = self.device
        net = self.net

        list_acc_class = list()
        list_acc_domain = list()
        list_loss_class = list()
        list_loss_domain = list()

        with torch.no_grad():
            for data, label in dataloader:
                domain_label = torch.ones_like(label) * domain_flag
                data, label, domain_label = data.to(device), label.to(device), domain_label.to(device)
                out_label, out_domain = net(data)

                loss_class = self.loss_fn_class(out_label, label)
                loss_domain = self.loss_fn_domain(out_domain, domain_label)

                list_loss_class.append(loss_class.cpu())
                list_loss_domain.append(loss_domain.cpu())

                out_label_index = torch.argmax(out_label, dim=-1)
                out_domain_index = torch.argmax(out_domain, dim=-1)

                list_acc_class.append(torch.eq(out_label_index, label).cpu())
                list_acc_domain.append(torch.eq(out_domain_index, domain_label).cpu())

        list_acc_domain = torch.cat(list_acc_domain)
        list_acc_class = torch.cat(list_acc_class)

        list_loss_class = torch.stack(list_loss_class)
        list_loss_domain = torch.stack(list_loss_domain)

        acc_class = torch.sum(list_acc_class) / torch.numel(list_acc_class)
        acc_domain = torch.sum(list_acc_domain) / torch.numel(list_acc_domain)

        loss_class = torch.mean(list_loss_class)
        loss_domain = torch.mean(list_loss_domain)

        return float(acc_class), float(acc_domain), float(loss_class), float(loss_domain)

    def eval_one_step(self):
        source_dataloader = self.source_dataloader
        target_dataloader = self.target_dataloader
        sr_acc_class, sr_acc_domain, sr_loss_class, sr_loss_domain = self.eval_one_dataloader(source_dataloader,
                                                                                              domain_flag=0)
        tg_acc_class, tg_acc_domain, tg_loss_class, tg_loss_domain = self.eval_one_dataloader(target_dataloader,
                                                                                              domain_flag=1)
        result_dict = {
            "sr_acc_class": sr_acc_class,
            "sr_acc_domain": sr_acc_domain,
            "sr_loss_class": sr_loss_class,
            "sr_loss_domain": sr_loss_domain,
            "tg_acc_class": tg_acc_class,
            "tg_acc_domain": tg_acc_domain,
            "tg_loss_class": tg_loss_class,
            "tg_loss_domain": tg_loss_domain
        }
        return result_dict


class TrainModuleBaseline(TrainModule):
    def __init__(self, device, args):
        super(TrainModuleBaseline, self).__init__(device, args)

        self.loss_fn_class = None

        self._init_output_dir("baseline")

    def set_module_parameters(self, net, loss_fn_class, optimizer, scheduler):
        self.net = net.to(self.device)
        self.loss_fn_class = loss_fn_class
        self.optimizer = optimizer
        self.scheduler = scheduler

    def eval_one_dataloader(self, dataloader):
        device = self.device
        net = self.net

        list_acc_class = list()
        list_loss_class = list()

        with torch.no_grad():
            for data, label in dataloader:
                data, label = data.to(device), label.to(device)
                out_label = net(data)

                loss_class = self.loss_fn_class(out_label, label)
                list_loss_class.append(loss_class.cpu())

                out_label_index = torch.argmax(out_label, dim=-1)

                list_acc_class.append(torch.eq(out_label_index, label).cpu())

        list_acc_class = torch.cat(list_acc_class)
        list_loss_class = torch.stack(list_loss_class)

        acc_class = torch.sum(list_acc_class) / torch.numel(list_acc_class)
        loss_class = torch.mean(list_loss_class)

        return float(acc_class), float(loss_class)

    def eval_one_step(self):
        source_dataloader = self.source_dataloader
        target_dataloader = self.target_dataloader
        sr_acc_class, sr_loss_class = self.eval_one_dataloader(source_dataloader)
        tg_acc_class, tg_loss_class = self.eval_one_dataloader(target_dataloader)
        result_dict = {
            "sr_acc_class": sr_acc_class,
            "sr_loss_class": sr_loss_class,
            "tg_acc_class": tg_acc_class,
            "tg_loss_class": tg_loss_class,
        }
        return result_dict

    def train_one_step(self, epoch: int = -1, total_epochs: int = 10):
        source_dataloader = self.source_dataloader
        net = self.net
        device = self.device

        steps_dataloader = len(source_dataloader)
        iter_source = iter(source_dataloader)

        for i in range(steps_dataloader):
            sr_data, sr_label = next(iter_source)
            sr_data, sr_label = sr_data.to(device), sr_label.to(device)
            sr_out_label = net(sr_data)

            loss_sr_class = self.loss_fn_class(sr_out_label, sr_label)

            self.optimizer.zero_grad()
            loss = loss_sr_class
            loss.backward()
            self.optimizer.step()
