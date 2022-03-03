import os
from utils.funcs import get_parser, seed_everything, ValueStat, get_timestamp, makedirs
from utils.funcs_plot import save_dataset
from Dataset.dataset_min_max import load_data

args = get_parser().parse_args()
seed_everything(args.seed)

if __name__ == '__main__':
    Output_root = "./Outputs/Problem3-min-max-visualization-mixed-data"
    Output_dir = os.path.join(Output_root,
                              f"Min-max-visualization_split-mode_{args.min_max_data_split_mode}_{get_timestamp()}")
    makedirs(Output_dir)
    min_max_shape = (3, 3)

    dataset_train, dataset_test = load_data(trainset_split_shape=min_max_shape,
                                            trainset_split_mode=args.min_max_data_split_mode)
    whole_dataset_train = dataset_train.get_whole_dataset()
    save_dataset(dataset=whole_dataset_train,
                 path=os.path.join(Output_dir, f'whole_dataset_train.jpg'),
                 )

    for _i in range(min_max_shape[0]):
        for _j in range(min_max_shape[1]):
            save_dataset(dataset=dataset_train[_i, _j],
                         path=os.path.join(Output_dir, f'data_{_i}_{_j}.jpg'),
                         )


