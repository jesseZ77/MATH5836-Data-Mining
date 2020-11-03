import numpy as np
from numpy import random
import math


class SimpleNetwork:
    def __init__(self, layers, layer_activations=None, learning_rate=0.01):
        self.layers = layers
        self.num_layers = len(layers)
        self.learning_rate = learning_rate
        self.weights = [np.random.normal(size=[layers[i], layers[i + 1]],
                                         scale=np.sqrt(2 / layers[i])) for i in range(len(layers) - 1)]
        self.biases = [np.random.normal(size=layers[i + 1],
                                        scale=np.sqrt(2 / layers[i])) for i in range(len(layers) - 1)]
        if layer_activations is None:
            layer_activations = ["sigmoid"] * (layers - 1)
        self.layer_activations = layer_activations

    @staticmethod
    def _act_func(x_vec, func):
        if func == "sigmoid":
            return 1 / (1 + np.exp(-x_vec))

        if func == "linear":
            return x_vec

        if func == "relu":
            return np.maximum(x_vec, 0)

    @staticmethod
    def _act_func_d(x_vec, func):
        if func == "sigmoid":
            return SimpleNetwork._act_func(x_vec, func) * (1 - SimpleNetwork._act_func(x_vec, func))

        if func == "linear":
            return 1

        if func == "relu":
            return np.where(x_vec > 0, 1, 0)

    @staticmethod
    def loss(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    @staticmethod
    def _loss_d(y_true, y_pred):
        return y_pred - y_true

    def _single_forward_pass(self, x, weights, biases, return_history=False):
        activations = [x]
        zs = []
        for i in range(self.num_layers - 1):
            zs.append(activations[-1].dot(weights[i]) + biases[i])
            activations.append(SimpleNetwork._act_func(zs[-1], self.layer_activations[i]))

        if return_history:
            return zs, activations
        return activations[-1]

    def predict(self, X, weights, biases):
        return np.array([self._single_forward_pass(x, weights, biases) for x in X])

    def _single_backward_pass(self, x, y, weights, biases):
        zs, activations = self._single_forward_pass(x, weights, biases, return_history=True)

        nabla_w = [np.zeros_like(weights[i]) for i in range(self.num_layers - 1)]
        nabla_b = [np.zeros_like(biases[i]) for i in range(self.num_layers - 1)]

        delta = SimpleNetwork._loss_d(y, activations[-1]) * \
                self._act_func_d(zs[-1], self.layer_activations[-1]) # output delta
        nabla_w[-1] = activations[-2].reshape(-1, 1).dot(delta.reshape(-1, 1).T)
        nabla_b[-1] = delta

        for i in range(self.num_layers - 2):
            delta = weights[-(i + 1)].dot(delta) * self._act_func_d(zs[-(i + 2)], self.layer_activations[-(i + 2)])
            nabla_w[-(i + 2)] = activations[-(i + 3)].reshape(-1, 1).dot(delta.reshape(-1, 1).T)
            nabla_b[-(i + 2)] = delta

        return nabla_w, nabla_b

    def compute_gradient(self, X, y, weights, biases):
        n = len(X)
        nabla_w = [np.zeros_like(self.weights[i]) for i in range(self.num_layers - 1)]
        nabla_b = [np.zeros_like(self.biases[i]) for i in range(self.num_layers - 1)]
        for x_vec, y_vec in zip(X, y):
            step_w, step_b = self._single_backward_pass(x_vec, y_vec, weights, biases)
            for i in range(self.num_layers - 1):
                nabla_w[i] += step_w[i] / n
                nabla_b[i] += step_b[i] / n
        return nabla_w, nabla_b

    def gradient_descent(self, X_train, y_train, X_test, y_test, n_epochs=100, update=True):
        train_loss = np.zeros([n_epochs])
        test_loss = np.zeros([n_epochs])

        w = self.weights.copy()
        b = self.biases.copy()
        for j in range(n_epochs):
            nabla_w, nabla_b = self.compute_gradient(X_train, y_train, w, b)
            for i in range(self.num_layers - 1):
                w[i] = w[i] - self.learning_rate * nabla_w[i]
                b[i] = b[i] - self.learning_rate * nabla_b[i]

            y_pred_train = self.predict(X_train, w, b)
            y_pred_test = self.predict(X_test, w, b)
            train_loss[j] = self.loss(y_train, y_pred_train)
            test_loss[j] = self.loss(y_test, y_pred_test)
            if (j+1) % 50 == 0:
                print(f"Epoch: {j+1}")
                print(f"train loss: {train_loss[j]}")
                print(f"test loss: {test_loss[j]}")
                print("\n")

        if update:
            self.weights = w
            self.biases = b

        return w, b, train_loss, test_loss


class BayesianNetwork(SimpleNetwork):
    def __init__(self, layers, layer_activations=None, learning_rate=0.01,
                 sigma2=25, nu1=0.01, nu2=0.01, use_langevin=False):
        super().__init__(layers, layer_activations, learning_rate)
        self.sigma2 = sigma2
        self.nu1 = nu1
        self.nu2 = nu2
        self.use_langevin = use_langevin
        self.tau2 = None

    def log_prior(self, weights, biases, tau2):
        part_1 = 0
        part_2 = 0

        for i in range(len(weights)):
            part_1 -= (weights[i].size + biases[i].size) * np.log(self.sigma2) / 2
            part_2 -= 0.5 * (np.sum(np.square(weights[i])) + np.sum(np.square(biases[i]))) / self.sigma2

        part3 = -(self.nu1 + 1) * np.log(tau2) - self.nu2 / tau2

        return part_1 + part_2 + part3

    def log_llk(self, y, X, weights, biases, tau2):
        y_pred = self.predict(X, weights, biases)
        part_1 = -len(y) / 2 * np.log(tau2)
        part_2 = -0.5 * np.sum(np.square(y_pred - y)) / tau2

        return part_1 + part_2

    def log_proposal(self, weights1, biases1, weights2, biases2, weight_prop_sd):
        output = 0

        for i in range(len(weights1)):
            output -= 0.5 * (np.sum(np.square(weights1[i] - weights2[i])) +
                             np.sum(np.square(biases1[i] - biases2[i]))) / weight_prop_sd ** 2

        return output

    def append_mc_sample(self, weights_list, biases_list, tau2_list, weights, biases, tau2, index):
        for i in range(self.num_layers - 1):
            weights_list[i][:, :, index] = weights[i].copy()
            biases_list[i][:, index] = biases[i].copy()
        tau2_list[index] = tau2
        return weights_list, biases_list, tau2_list

    def mcmc_sampler(self, X_train, y_train, X_test, y_test,
                     n_samples=100, weight_prop_sd=0.025, tau_prop_sd=0.001, update=True, burn_in=0.25):

        train_loss = np.zeros([n_samples])
        test_loss = np.zeros([n_samples])
        accepted = np.zeros([n_samples])
        accepted[0] = 1

        w_k = self.weights.copy()
        b_k = self.biases.copy()


        y_pred_train = self.predict(X_train, w_k, b_k)
        y_pred_test = self.predict(X_test, w_k, b_k)
        train_loss[0] = self.loss(y_train, y_pred_train)
        test_loss[0] = self.loss(y_test, y_pred_test)

        if self.tau2 is None:
            self.tau2 = np.var(y_pred_train - y_train)
        tau2_k = self.tau2

        w_samples = [np.zeros(list(w.shape) + [n_samples]) for w in self.weights]
        b_samples = [np.zeros(list(b.shape) + [n_samples]) for b in self.biases]
        tau2_samples = np.zeros(n_samples)

        w_samples, b_samples, tau2_samples = self.append_mc_sample(w_samples, b_samples, tau2_samples,
                                                                   w_k, b_k, tau2_k, index=0)

        for j in range(n_samples - 1):
            if self.use_langevin:
                nabla_w_k, nabla_b_k = self.compute_gradient(X_train, y_train, w_k, b_k)
                w_bar_k = [w_k[i] - self.learning_rate * nabla_w_k[i] for i in range(self.num_layers - 1)]
                b_bar_k = [b_k[i] - self.learning_rate * nabla_b_k[i] for i in range(self.num_layers - 1)]

                w_p = [w + np.random.normal(size=w.shape, scale=weight_prop_sd) for w in w_bar_k]
                b_p = [b + np.random.normal(size=b.shape, scale=weight_prop_sd) for b in b_bar_k]

                nabla_w_p, nabla_b_p = self.compute_gradient(X_train, y_train, w_p, b_p)
                w_bar_p = [w_p[i] - self.learning_rate * nabla_w_p[i] for i in range(self.num_layers - 1)]
                b_bar_p = [b_p[i] - self.learning_rate * nabla_b_p[i] for i in range(self.num_layers - 1)]
            else:
                w_p = [w + np.random.normal(size=w.shape, scale=weight_prop_sd) for w in w_k]
                b_p = [b + np.random.normal(size=b.shape, scale=weight_prop_sd) for b in b_k]

            tau2_p = tau2_k + np.random.normal(size=1, scale=tau_prop_sd)

            num_part_1 = self.log_llk(y_train, X_train, w_p, b_p, tau2_p)
            num_part_2 = self.log_prior(w_p, b_p, tau2_p)
            num_part_3 = self.log_proposal(w_k, b_k, w_bar_p, b_bar_p, weight_prop_sd)

            alpha_num_log = num_part_1 + num_part_2 + num_part_3

            den_part_1 = self.log_llk(y_train, X_train, w_k, b_k, tau2_k)
            den_part_2 = self.log_prior(w_k, b_k, tau2_k)
            den_part_3 = self.log_proposal(w_p, b_p, w_bar_k, b_bar_k, weight_prop_sd)

            alpha_den_log = den_part_1 + den_part_2 + den_part_3

            try:
                alpha = math.exp(alpha_num_log - alpha_den_log)
            except OverflowError as e:
                alpha = 1

            if random.uniform(size=1) < alpha:
                w_k = w_p.copy()
                b_k = b_p.copy()
                tau2_k = tau2_p
                accepted[j] = 1

            w_samples, b_samples, tau2_samples = self.append_mc_sample(w_samples, b_samples, tau2_samples,
                                                                       w_k, b_k, tau2_k, index=j)
            y_pred_train = self.predict(X_train, w_k, b_k)
            y_pred_test = self.predict(X_test, w_k, b_k)
            train_loss[j] = self.loss(y_train, y_pred_train)
            test_loss[j] = self.loss(y_test, y_pred_test)

            if (j + 2) % 50 == 0:
                print(f"Epoch: {j+2}")
                print(f"train loss: {train_loss[j]}")
                print(f"test loss: {test_loss[j]}")
                print("\n")

        if update:
            for i in range(self.num_layers - 1):
                self.weights[i] = np.mean(w_samples[i][:, :, int(burn_in * n_samples):], axis=2)
                self.biases[i] = np.mean(b_samples[i][:, int(burn_in * n_samples):], axis=1)

        return w_samples, b_samples, tau2_samples, train_loss, test_loss, accepted