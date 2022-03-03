import os.path
import time
from typing import Union

import numpy as np
from tqdm import tqdm

from Dataset import dataset
from Dataset.dataset_min_max import load_data
from Model import Module
from Model.Loss import MSELoss
from Model.MLQP import MinMaxModule
from Model.Optim import SGD, Optimizer
from utils.funcs import get_parser, seed_everything, ValueStat, get_timestamp, makedirs
from utils.funcs_plot import save_dataset_model_plot, create_gif_file
from utils.funcs_stat import ValuesVisual

args = get_parser().parse_args()
seed_everything(args.seed)


def train(_data_loader: dataset.Dataloader = None, _model: Module.Network = None,
          _loss_fn: Module.Network = None, _optimizer: Optimizer = None, **kwargs):
    _model.train()
    loss_stat = ValueStat()

    for iter, (data, label) in enumerate(_data_loader):
        output = _model(data)
        loss = _loss_fn(output, label)

        _optimizer.zero_grad()
        _model.backward(_loss_fn.backward())
        _optimizer.step()

        loss_stat.update(loss)
    return {'loss': loss_stat.get_avg()}


def test(_data_loader: dataset.Dataloader = None, _model: Union[Module.Network, MinMaxModule] = None, **kwargs):
    _model.eval()
    acc_stat = ValueStat()
    for iter, (data, label) in enumerate(_data_loader):
        output = _model(data)
        predict = np.where(output > 0.5, 1, 0)
        judgement = np.where(predict == label, 1, 0)
        acc_stat.update(np.sum(judgement) / np.size(judgement))
    return {'accuracy': acc_stat.get_avg()}


def run_min_max(_dataset_train: dataset.Dataset, _dataset_test: dataset.Dataset,
                _model: Module.Network = None, _output_root: str = './'):
    log_file_path = os.path.join(_output_root, "run.log")
    Visualization_dir = os.path.join(_output_root, "Visualization")
    Image_train_dir = os.path.join(Visualization_dir, "train_data_image")
    Image_test_dir = os.path.join(Visualization_dir, "test_data_image")
    makedirs(Image_train_dir)
    makedirs(Image_test_dir)

    dataloader_train = dataset.Dataloader(dataset=_dataset_train, shuffle=True, batch_size=1)
    dataloader_test = dataset.Dataloader(dataset=_dataset_test, shuffle=False, batch_size=1)

    loss_fn = MSELoss()
    optimizer = SGD(network=_model, lr=args.lr)

    loss_vv = ValuesVisual()
    acc_vv = ValuesVisual()

    start_time = time.time()

    for epoch in range(args.Epochs + 1):

        train_result = train(dataloader_train, _model, loss_fn, optimizer, epoch=epoch)

        loss_vv.add(train_result['loss'])
        Output = "[Train] epoch={:0>4d}, loss={:.4f}, time={:.3f}s".format(epoch, train_result['loss'],
                                                                           time.time() - start_time
                                                                           )
        # print(Output)
        with open(log_file_path, 'a') as file:
            file.write(Output + '\n')

        test_result = test(dataloader_test, _model, epoch=epoch)

        acc_vv.add(test_result['accuracy'])
        Output = "[Test] epoch={:0>4d}, accuracy={:.2%}, time={:.3f}s\n".format(epoch, test_result['accuracy'],
                                                                                time.time() - start_time
                                                                                )
        # print(Output)
        with open(log_file_path, 'a') as file:
            file.write(Output)

        if epoch % 20 == 0:
            save_dataset_model_plot(dataset=_dataset_train, model=_model,
                                    save_path=os.path.join(Image_train_dir, "epoch_{:0>4d}".format(epoch)),
                                    title="Iteration: {:>8d}".format(epoch * len(dataloader_train)),
                                    label_neg="Negative (train)", label_pos="Positive (train)"
                                    )
            save_dataset_model_plot(dataset=_dataset_test, model=_model,
                                    save_path=os.path.join(Image_test_dir, "epoch_{:0>4d}".format(epoch)),
                                    title="Iteration: {:>8d}".format(epoch * len(dataloader_train)),
                                    label_neg="Negative (test)", label_pos="Positive (test)"
                                    )

            loss_vv.plot(output_path=os.path.join(Visualization_dir, "loss_curve.jpg"),
                         title="Loss curve", xlabel="Iteration", ylabel="Loss", duration=len(dataloader_train)
                         )
            acc_vv.plot(output_path=os.path.join(Visualization_dir, "accuracy_curve.jpg"),
                        title="Accuracy curve", xlabel="Iteration", ylabel="Accuracy", duration=len(dataloader_train)
                        )
        if epoch % 50 == 0:
            create_gif_file(image_dir=Image_train_dir, gif_path=os.path.join(Visualization_dir, "data.gif"),
                            duration=0.8)


def test_min_max_module(_dataset_train: dataset.Dataset, _dataset_test: dataset.Dataset,
                        _model: MinMaxModule, _output_root: str = './'):
    log_file_path = os.path.join(_output_root, "run.log")

    dataloader_test = dataset.Dataloader(dataset=_dataset_test, shuffle=False, batch_size=1)
    test_result = test(dataloader_test, _model)

    Output = "[Test-Min-Max Final-test dataset] accuracy={:.2%}".format(test_result['accuracy'])
    print(Output)
    with open(log_file_path, 'a') as file:
        file.write(Output + '\n')

    save_dataset_model_plot(dataset=_dataset_test, model=_model,
                            save_path=os.path.join(_output_root, "decision_boundary-testset.jpg"),
                            title=f"Min-Max Decision Boundary-Total Iteration:{args.Epochs*len(dataset_train)}",
                            label_neg="Negative (test)", label_pos="Positive (test)"
                            )

    dataloader_train = dataset.Dataloader(dataset=_dataset_train, shuffle=False, batch_size=1)
    train_result = test(dataloader_train, _model)

    Output = "[Test-Min-Max Final-train dataset] accuracy={:.2%}".format(train_result['accuracy'])
    print(Output)
    with open(log_file_path, 'a') as file:
        file.write(Output + '\n')

    save_dataset_model_plot(dataset=_dataset_train, model=_model,
                            save_path=os.path.join(_output_root, "decision_boundary-trainset.jpg"),
                            title=f"Min-Max Decision Boundary-Total Iteration:{args.Epochs*len(dataset_train)}",
                            label_neg="Negative (train)", label_pos="Positive (train)"
                            )


if __name__ == '__main__':
    Output_root = "./Outputs/Problem3-min-max"
    Output_dir = os.path.join(Output_root,
                              f"MLQP-min-max_split-mode_{args.min_max_data_split_mode}_lr_{args.lr}_{get_timestamp()}")
    makedirs(Output_dir)
    min_max_shape = (3, 3)

    dataset_train, dataset_test = load_data(trainset_split_shape=min_max_shape,
                                            trainset_split_mode=args.min_max_data_split_mode)
    # whole_dataset_train = dataset_train.get_whole_dataset()

    min_max_module = MinMaxModule(model_name="MultilayerQuadraticPerceptron", shape=min_max_shape, hidden_size=[100])
    with tqdm(total=min_max_shape[0] * min_max_shape[1]) as run_bar:
        for _i in range(min_max_shape[0]):
            for _j in range(min_max_shape[1]):
                run_bar.set_postfix({"running sub-problem": f"({_i}, {_j})"})
                
                Output_min_max_root = os.path.join(Output_dir, f"min_max_{_i}_{_j}")
                makedirs(Output_min_max_root)

                run_min_max(dataset_train[_i, _j], dataset_test, min_max_module[_i, _j], Output_min_max_root)

                run_bar.update(1)
                
    test_min_max_module(dataset_train.get_whole_dataset(), dataset_test, min_max_module, Output_dir)
