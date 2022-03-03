import matplotlib.pyplot as plt


class ValuesVisual:
    def __init__(self):
        self.values = []

    def add(self, val):
        self.values.append(val)

    def __len__(self):
        return len(self.values)

    def plot(self, output_path, title=None, xlabel=None, ylabel=None, duration=1):
        length = len(self)
        if length == 0:
            return -1
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x_axis = [i*duration for i in range(length)]
        ax.plot(x_axis, self.values, color='tab:blue')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.savefig(output_path)
        plt.close()