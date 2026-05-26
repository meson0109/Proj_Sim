"""
torch.nn里面写了已经封装好的训练函数,这里是我重新拆出来写的本地实现
"""

import copy
import time

import torch


def softmax_loss(x, y):
    shifted = x - x.max(dim=1, keepdim=True).values
    log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
    loss = -log_probs[torch.arange(x.shape[0], device=x.device), y].mean()
    dx = torch.exp(log_probs)
    dx[torch.arange(x.shape[0], device=x.device), y] -= 1
    dx /= x.shape[0]
    return loss, dx


class Linear:
    @staticmethod
    def forward(x, w, b):
        x_flat = x.reshape(x.shape[0], -1)
        out = x_flat.mm(w) + b
        return out, (x, w, b)

    @staticmethod
    def backward(dout, cache):
        x, w, _ = cache
        x_flat = x.reshape(x.shape[0], -1)
        dx = dout.mm(w.t()).reshape_as(x)
        dw = x_flat.t().mm(dout)
        db = dout.sum(dim=0)
        return dx, dw, db


class ReLU:
    @staticmethod
    def forward(x):
        return torch.clamp(x, min=0), x

    @staticmethod
    def backward(dout, cache):
        return dout * (cache > 0)


class LinearReLU:
    @staticmethod
    def forward(x, w, b):
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        return out, (fc_cache, relu_cache)

    @staticmethod
    def backward(dout, cache):
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        return Linear.backward(da, fc_cache)


class FastConv:
    @staticmethod
    def forward(x, w, b, conv_param):
        _, c, _, _ = x.shape
        f, _, hh, ww = w.shape
        stride, pad = conv_param["stride"], conv_param["pad"]
        layer = torch.nn.Conv2d(c, f, (hh, ww), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        return out, (tx, out, layer)

    @staticmethod
    def backward(dout, cache):
        tx, out, layer = cache
        try:
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
            dw = torch.zeros_like(layer.weight)
            db = torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool:
    @staticmethod
    def forward(x, pool_param):
        layer = torch.nn.MaxPool2d(
            kernel_size=(pool_param["pool_height"], pool_param["pool_width"]),
            stride=pool_param["stride"],
        )
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        return out, (tx, out, layer)

    @staticmethod
    def backward(dout, cache):
        tx, out, layer = cache
        try:
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class ConvReLUPool:
    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        return out, (conv_cache, relu_cache, pool_cache)

    @staticmethod
    def backward(dout, cache):
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        return FastConv.backward(da, conv_cache)


class ThreeLayerConvNet:
    """conv - relu - 2x2 pool - linear - relu - linear - softmax."""

    def __init__(
        self,
        input_dims=(1, 28, 28),
        num_filters=32,
        filter_size=5,
        hidden_dim=128,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=torch.float32,
        device="cpu",
    ):
        c, h, w = input_dims
        pooled_h, pooled_w = h // 2, w // 2
        self.params = {
            "W1": weight_scale
            * torch.randn(num_filters, c, filter_size, filter_size, dtype=dtype, device=device),
            "b1": torch.zeros(num_filters, dtype=dtype, device=device),
            "W2": weight_scale
            * torch.randn(num_filters * pooled_h * pooled_w, hidden_dim, dtype=dtype, device=device),
            "b2": torch.zeros(hidden_dim, dtype=dtype, device=device),
            "W3": weight_scale * torch.randn(hidden_dim, num_classes, dtype=dtype, device=device),
            "b3": torch.zeros(num_classes, dtype=dtype, device=device),
        }
        self.reg = reg
        self.dtype = dtype

    def loss(self, x, y=None):
        x = x.to(self.dtype)
        w1, b1 = self.params["W1"], self.params["b1"]
        w2, b2 = self.params["W2"], self.params["b2"]
        w3, b3 = self.params["W3"], self.params["b3"]
        conv_param = {"stride": 1, "pad": (w1.shape[2] - 1) // 2}
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        out1, cache1 = ConvReLUPool.forward(x, w1, b1, conv_param, pool_param)
        out2, cache2 = LinearReLU.forward(out1, w2, b2)
        scores, cache3 = Linear.forward(out2, w3, b3)
        if y is None:
            return scores

        loss, dscores = softmax_loss(scores, y)
        loss += self.reg * (torch.sum(w1 * w1) + torch.sum(w2 * w2) + torch.sum(w3 * w3))
        dout2, dw3, db3 = Linear.backward(dscores, cache3)
        dout1, dw2, db2 = LinearReLU.backward(dout2, cache2)
        _, dw1, db1 = ConvReLUPool.backward(dout1, cache1)
        grads = {
            "W1": dw1 + 2 * self.reg * w1,
            "b1": db1,
            "W2": dw2 + 2 * self.reg * w2,
            "b2": db2,
            "W3": dw3 + 2 * self.reg * w3,
            "b3": db3,
        }
        return loss, grads


def adam(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", torch.zeros_like(w))
    config.setdefault("v", torch.zeros_like(w))
    config.setdefault("t", 0)
    config["t"] += 1
    config["m"] = config["beta1"] * config["m"] + (1 - config["beta1"]) * dw
    config["v"] = config["beta2"] * config["v"] + (1 - config["beta2"]) * dw * dw
    m_hat = config["m"] / (1 - config["beta1"] ** config["t"])
    v_hat = config["v"] / (1 - config["beta2"] ** config["t"])
    next_w = w - config["learning_rate"] * m_hat / (torch.sqrt(v_hat) + config["epsilon"])
    return next_w, config


class Solver:
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]
        self.update_rule = kwargs.pop("update_rule", adam)
        self.optim_config = kwargs.pop("optim_config", {"learning_rate": 1e-3})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 128)
        self.num_epochs = kwargs.pop("num_epochs", 5)
        self.device = kwargs.pop("device", "cpu")
        self.print_every = kwargs.pop("print_every", 50)
        self.verbose = kwargs.pop("verbose", True)
        if kwargs:
            raise ValueError(f"Unknown Solver arguments: {sorted(kwargs)}")
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.best_val_acc = 0.0
        self.best_params = {}
        self.optim_configs = {k: copy.copy(self.optim_config) for k in self.model.params}

    def _step(self):
        mask = torch.randperm(self.X_train.shape[0])[: self.batch_size]
        x_batch = self.X_train[mask].to(self.device)
        y_batch = self.y_train[mask].to(self.device)
        loss, grads = self.model.loss(x_batch, y_batch)
        self.loss_history.append(float(loss.item()))
        with torch.no_grad():
            for name, value in self.model.params.items():
                next_value, next_config = self.update_rule(
                    value, grads[name], self.optim_configs[name]
                )
                self.model.params[name] = next_value
                self.optim_configs[name] = next_config

    def check_accuracy(self, x, y, batch_size=256):
        x = x.to(self.device)
        y = y.to(self.device)
        preds = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                scores = self.model.loss(x[start : start + batch_size])
                preds.append(scores.argmax(dim=1))
        pred = torch.cat(preds)
        return float((pred == y).float().mean().item())

    def train(self):
        iterations_per_epoch = max(self.X_train.shape[0] // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch
        start_time = time.time()
        for t in range(num_iterations):
            self._step()
            if self.verbose and (t + 1) % self.print_every == 0:
                elapsed = time.time() - start_time
                print(f"iter {t + 1}/{num_iterations}, loss {self.loss_history[-1]:.4f}, {elapsed:.1f}s")
            if (t + 1) % iterations_per_epoch == 0:
                for config in self.optim_configs.values():
                    config["learning_rate"] *= self.lr_decay
                train_acc = self.check_accuracy(self.X_train, self.y_train)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                if self.verbose:
                    print(f"epoch {(t + 1) // iterations_per_epoch}: train {train_acc:.4f}, val {val_acc:.4f}")
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {k: v.clone() for k, v in self.model.params.items()}
        if self.best_params:
            self.model.params = self.best_params

