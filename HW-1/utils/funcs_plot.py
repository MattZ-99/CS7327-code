import os
import math
import matplotlib.pyplot as plt
import numpy as np
import imageio

COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (1, 1, 1)


def plot_dataset(dataset):
    features = dataset.features
    labels = dataset.labels

    features_0, features_1 = features[labels == 0], features[labels == 1]
    labels_0, labels_1 = labels[labels == 0], labels[labels == 1]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(*features_0.T, label="Negative")
    ax.scatter(*features_1.T, label="Positive")
    # plt.tight_layout()
    plt.legend()
    plt.show()


def save_dataset(dataset, path='./', plot_pos=True, plot_neg=True):
    features = dataset.features
    labels = dataset.labels

    features_0, features_1 = features[labels == 0], features[labels == 1]
    labels_0, labels_1 = labels[labels == 0], labels[labels == 1]

    fig, ax = plt.subplots(figsize=(10, 10))

    if plot_neg:
        ax.scatter(*features_0.T, label="Negative", color="C0")
    if plot_pos:
        ax.scatter(*features_1.T, label="Positive", color="C1")
    # plt.tight_layout()
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    plt.legend()
    plt.savefig(path)
    plt.close('all')


def plot_dataset_together(trainset, testset):
    train_features, train_labels = trainset.features, trainset.labels
    test_features, test_labels = testset.features, testset.labels

    train_features_0, train_features_1 = train_features[train_labels == 0], train_features[train_labels == 1]
    test_features_0, test_features_1 = test_features[test_labels == 0], test_features[test_labels == 1]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(*train_features_0.T, label="Train-Negative")
    ax.scatter(*train_features_1.T, label="Train-Positive")
    ax.scatter(*test_features_0.T, label="Test-Negative")
    ax.scatter(*test_features_1.T, label="Test-Positive")
    # plt.tight_layout()
    plt.legend()
    plt.show()


def calculate_contourf_val(model, left=-6, right=6.1, step=0.05):
    grid_data = np.mgrid[left:right:step, left:right:step]
    grid_data = grid_data.transpose((1, 2, 0))
    val = model(grid_data)

    return grid_data[:, :, 0], grid_data[:, :, 1], np.squeeze(val)


def plot_dataset_model(dataset, model):
    features = dataset.features
    labels = dataset.labels

    contourf_x, contourf_y, contourf_val = calculate_contourf_val(model)

    features_0, features_1 = features[labels == 0], features[labels == 1]
    labels_0, labels_1 = labels[labels == 0], labels[labels == 1]

    fig, ax = plt.subplots(figsize=(10, 10))
    cs = ax.contourf(contourf_x, contourf_y, contourf_val, levels=[0, 0.5, 1])
    ax.scatter(*features_0.T, label="Negative")
    ax.scatter(*features_1.T, label="Positive")
    cbar = fig.colorbar(cs)
    # plt.tight_layout()
    plt.legend()
    plt.show()


def save_dataset_model_plot(dataset, model, save_path, *args, **kwargs):
    features = dataset.features
    labels = dataset.labels

    contourf_x, contourf_y, contourf_val = calculate_contourf_val(model)

    features_0, features_1 = features[labels == 0], features[labels == 1]
    labels_0, labels_1 = labels[labels == 0], labels[labels == 1]

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="grey")
    cs_levels = [0., 0.2, 0.4, 0.6, 0.8, 1.]
    cs_colors = [(*COLOR_BLACK, 1.), (*COLOR_BLACK, 0.6), (*COLOR_BLACK, 0.3), (*COLOR_BLACK, 0.1), (*COLOR_BLACK, 0.),
                 ]
    cs = ax.contourf(contourf_x, contourf_y, contourf_val, levels=cs_levels, colors=cs_colors)
    cbar = fig.colorbar(cs)

    if "label_neg" in kwargs:
        ax.scatter(*features_0.T, label=kwargs["label_neg"])
    if "label_pos" in kwargs:
        ax.scatter(*features_1.T, label=kwargs["label_pos"])
    if "title" in kwargs:
        ax.set_title(kwargs["title"], fontsize=15)

    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-6.5, 6.5)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def create_gif_file(image_dir, gif_path, duration=0.1, max_flame=50, verbose=0):
    image_list = list()
    for image_file in sorted(os.listdir(image_dir)):
        image_list.append(os.path.join(image_dir, image_file))
    # _interval = math.ceil(len(image_list) / max_flame)
    # image_list = [image_list[_i*_interval] for _i in range(len(image_list)//_interval)]
    _create_gif(image_list, gif_path, duration, verbose)


def _create_gif(image_list, gif_name, duration=0.1, verbose=0):
    frames = []
    for image_name in image_list:
        if verbose == 1:
            print(image_name)
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return
