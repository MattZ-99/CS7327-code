import os.path
import time
import numpy as np
from utils.funcs import get_parser, seed_everything, ValueStat, get_timestamp, makedirs
from utils.funcs_plot import save_dataset_model_plot, create_gif_file
from utils.funcs_stat import ValuesVisual
from Dataset import dataset
from Model.MLQP import MultilayerQuadraticPerceptron
from Model.Loss import MSELoss
from Model.Optim import SGD
from tqdm import tqdm

args = get_parser().parse_args()
seed_everything(args.seed)

if __name__ == "__main__":
    Output_root = "./Outputs/Problem2"
    Output_dir = os.path.join(Output_root, f"MLQP_lr_{args.lr}_{get_timestamp()}")
    Visualization_dir = os.path.join(Output_dir, "Visualization")
    Image_train_dir = os.path.join(Visualization_dir, "train_data_image")
    Image_test_dir = os.path.join(Visualization_dir, "test_data_image")
    log_file_path = os.path.join(Output_dir, "run.log")
    makedirs(Image_train_dir)
    makedirs(Image_test_dir)

    dataset_train, dataset_test = dataset.load_data()
    dataloader_train = dataset.Dataloader(dataset=dataset_train, shuffle=True, batch_size=1)
    dataloader_test = dataset.Dataloader(dataset=dataset_test, shuffle=False, batch_size=1)

    model = MultilayerQuadraticPerceptron(hidden_size=[100])
    loss_fn = MSELoss()
    optimizer = SGD(network=model, lr=args.lr)


    loss_vv = ValuesVisual()
    acc_vv = ValuesVisual()
    loss_stat = ValueStat()
    acc_stat = ValueStat()

    start_time = time.time()

    for epoch in range(args.Epochs+1):
        model.train()
        loss_stat.reset()
        run_bar = tqdm(enumerate(dataloader_train), desc=f"[Train] Epoch={epoch}", total=len(dataloader_train))
        for iter, (data, label) in run_bar:
            output = model(data)
            loss = loss_fn(output, label)

            optimizer.zero_grad()
            model.backward(loss_fn.backward())
            optimizer.step()

            loss_stat.update(loss)
            run_bar.set_postfix({'loss': loss_stat.get_avg()})

        loss_vv.add(loss_stat.get_avg())
        Output = "[Train] epoch={:0>4d}, loss={:.4f}, time={:.3f}s\n".format(epoch, loss_stat.get_avg(),
                                                                         time.time()-start_time
                                                                         )
        print(Output)
        with open(log_file_path, 'a') as file:
            file.write(Output)

        model.eval()
        acc_stat.reset()
        run_bar = tqdm(enumerate(dataloader_test), desc=f"[Test] Epoch={epoch}", total=len(dataloader_test))
        for iter, (data, label) in run_bar:
            output = model(data)
            predict = np.where(output > 0.5, 1, 0)
            judgement = np.where(predict == label, 1, 0)
            acc_stat.update(np.sum(judgement) / np.size(judgement))
            run_bar.set_postfix({"accuracy": acc_stat.get_avg()})

        acc_vv.add(acc_stat.get_avg())
        Output = "[Test] epoch={:0>4d}, accuracy={:.2%}, time={:.3f}s\n".format(epoch, acc_stat.get_avg(),
                                                                            time.time()-start_time
                                                                            )
        print(Output)
        with open(log_file_path, 'a') as file:
            file.write(Output)

        if epoch % 20 == 0:
            save_dataset_model_plot(dataset=dataset_train, model=model,
                                    save_path=os.path.join(Image_train_dir, "epoch_{:0>4d}".format(epoch)),
                                    title="Iteration: {:>8d}".format(epoch * len(dataloader_train)),
                                    label_neg="Negative (train)", label_pos="Positive (train)"
                                    )
            save_dataset_model_plot(dataset=dataset_test, model=model,
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
            create_gif_file(image_dir=Image_train_dir, gif_path=os.path.join(Visualization_dir, "data.gif"), duration=0.8)
        