import sys

from .Module import Network, Parameters, ModuleList
import numpy as np


class SinglelayerQuadraticPerceptron(Network):
    def __init__(self, input_size=2, output_size=1, bias=True):
        super(SinglelayerQuadraticPerceptron, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight0 = Parameters((input_size, output_size))
        self.weight1 = Parameters((input_size, output_size))
        if bias:
            self.bias = Parameters(output_size)

        self.parameters_list = ["weight0", "weight1", "bias"]
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight0.data = np.random.uniform(-1, 1, size=self.weight0.data.shape)
        self.weight1.data = np.random.uniform(-1, 1, size=self.weight1.data.shape)
        if self.bias is not None:
            self.bias.data = np.random.uniform(-1, 1, size=self.bias.data.shape)

    def forward(self, x):
        xu2 = np.multiply(x, x)
        self.buffer["xu2"] = xu2
        self.buffer["x"] = x
        return np.dot(xu2, self.weight0.data) + np.dot(x, self.weight1.data) + self.bias.data

    def backward(self, gradient):
        self.bias.gradient += np.sum(gradient, axis=0) * 1
        self.weight0.gradient += np.sum(np.matmul(np.expand_dims(self.buffer["xu2"], axis=-1),
                                                  np.expand_dims(gradient, axis=1)), axis=0)
        self.weight1.gradient += np.sum(np.matmul(np.expand_dims(self.buffer["x"], axis=-1),
                                                  np.expand_dims(gradient, axis=1)), axis=0)

        gradient_new = 2 * np.multiply(np.repeat(np.expand_dims(self.buffer["x"], axis=-1), self.output_size, axis=-1),
                                       np.repeat(np.expand_dims(self.weight0.data, axis=0), self.buffer["x"].shape[0],
                                                 axis=0)) + np.repeat(np.expand_dims(self.weight1.data, axis=0),
                                                                      self.buffer["x"].shape[0], axis=0)

        gradient_new = np.matmul(gradient_new, np.expand_dims(gradient, axis=-1))
        return np.squeeze(gradient_new)


class Sigmoid(Network):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        x = 1 / (1 + np.exp(-x))
        self.buffer['f_x'] = x
        return x

    def backward(self, gradient):
        return np.multiply(gradient, np.multiply(self.buffer['f_x'], 1 - self.buffer['f_x']))


class MultilayerQuadraticPerceptron(Network):
    def __init__(self, input_size: int = 2, output_size=1, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = [10]

        self.SLQP_0 = SinglelayerQuadraticPerceptron(input_size=input_size, output_size=hidden_size[0])
        self.sigmoid_0 = Sigmoid()
        self.hidden_layer_list = ModuleList()
        for _i in range(1, len(hidden_size)):
            self.hidden_layer_list.append(
                SinglelayerQuadraticPerceptron(input_size=hidden_size[_i - 1], output_size=hidden_size[_i])
            )
            self.hidden_layer_list.append(
                Sigmoid()
            )
        self.SLQP_1 = SinglelayerQuadraticPerceptron(input_size=hidden_size[-1], output_size=output_size)
        self.sigmoid_1 = Sigmoid()

        self._make_parameters()

    def _make_parameters(self):
        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, Network):
                if isinstance(attribute, ModuleList):
                    for _id, layer in enumerate(attribute):
                        self.parameters_list.extend(
                            ['.'.join((f"{attribute_name}.__ModuleList__.{_id}", para)) for para in
                             layer.parameters_list])
                else:
                    self.parameters_list.extend(
                        ['.'.join((attribute_name, para)) for para in attribute.parameters_list])

    def forward(self, x):
        x = self.SLQP_0(x)
        x = self.sigmoid_0(x)
        for layer in self.hidden_layer_list:
            x = layer(x)
        x = self.SLQP_1(x)
        x = self.sigmoid_1(x)

        return x

    def backward(self, gradient):
        gradient = self.sigmoid_1.backward(gradient)
        gradient = self.SLQP_1.backward(gradient)
        for layer in self.hidden_layer_list[::-1]:
            gradient = layer.backward(gradient)
        gradient = self.sigmoid_0.backward(gradient)
        gradient = self.SLQP_0.backward(gradient)

        return gradient


class MinMaxModule(object):
    def __init__(self, model_name=None, shape=None, **model_parameters):
        super().__init__()
        self.model_list = None
        self.shape = shape
        self.model_name = model_name
        self._init_module_list(**model_parameters)
        self._training = False

    def _init_module_list(self, **model_parameters):
        if self.model_name == "MultilayerQuadraticPerceptron":
            model_class = getattr(sys.modules[__name__], self.model_name)
            if "hidden_size" in model_parameters:
                hidden_size = model_parameters["hidden_size"]
            else:
                hidden_size = None
            model_list = list()
            for _i in range(self.shape[0]):
                ml = list()
                for _j in range(self.shape[1]):
                    ml.append(model_class(hidden_size=hidden_size))
                model_list.append(ml)
            self.model_list = model_list

        else:
            raise NotImplementedError

    def __getitem__(self, item):
        return self.model_list[item[0]][item[1]]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def eval(self):
        self._training = False

    def forward(self, item):
        output = list()
        for _i in range(self.shape[0]):
            out = list()
            for _j in range(self.shape[1]):
                o = self.model_list[_i][_j](item)
                out.append(o)
            out = np.stack(out, axis=0)
            output.append(out)
        output = np.stack(output, axis=0)
        output = np.min(output, axis=1)
        output = np.max(output, axis=0)
        return output


