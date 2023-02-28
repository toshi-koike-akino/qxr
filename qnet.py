# QNN models
# (c) 2023 Toshiaki Koike-Akino

import pennylane as qml
import torch
import numpy as np


# QNN: Angle / 2-Design / Z
class QNN(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        nlayer=2,
        qdev="default.qubit",
        emb="amp",
        meas="probs",
    ):
        super(QNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nlayer = nlayer
        self.qdev = qdev
        self.emb = emb
        self.meas = meas

        self.set_qubit()
        print("qubit", self.qubit)

        self.wires = range(self.qubit)
        self.dev = qml.device(qdev, wires=self.qubit)
        print("qdev", self.dev)

        self.qlayer = self.get_qlayer()
        print("qlayer", self.qlayer)

    def set_qubit(self):
        if self.emb == "angle":
            isize = self.in_features
        else:
            isize = int(np.ceil(np.log2(self.in_features)))

        if self.meas == "expval":
            osize = self.out_features
        else:
            osize = int(np.ceil(np.log2(self.out_features)))

        self.qubit = np.max([isize, osize, 1])

    # main VQC
    def qcircuit(self, inputs, weights, bias):
        if self.emb == "angle":
            qml.AngleEmbedding(inputs, wires=self.wires)
        else:
            qml.AmplitudeEmbedding(inputs, wires=self.wires, pad_with=1, normalize=True)
        qml.SimplifiedTwoDesign(bias, weights, wires=self.wires)
        if self.meas == "expval":
            return [qml.expval(qml.PauliZ(k)) for k in self.wires]
        else:
            return qml.probs()

    # trainable weights shape: [bias, weights]
    def get_shapes(self):
        self.shapes = qml.SimplifiedTwoDesign.shape(
            n_layers=self.nlayer, n_wires=self.qubit
        )

        print("shapes", self.shapes)
        return self.shapes

    # wrap to torch layer
    def get_qlayer(self):
        qnode = qml.QNode(self.qcircuit, device=self.dev)
        shapes = self.get_shapes()
        shape = {"bias": shapes[0], "weights": shapes[1]}
        qlayer = qml.qnn.TorchLayer(qnode, shape)
        return qlayer

    def forward(self, input):
        output = self.qlayer(input)
        output = output[..., -self.out_features :]  # truncate heads
        return output


# Quanv2d
class Quanv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        nlayer=2,
        qdev="default.qubit",
        emb="amp",
        meas="probs",
    ):
        super(Quanv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # 2D padding to be same size
        pad_left = (kernel_size - 1) // 2
        pad_right = kernel_size // 2
        self.pad = (pad_left, pad_right, pad_left, pad_right)

        # QNN
        self.nlayer = nlayer
        self.qdev = qdev

        in_features = in_channels * self.kernel_size**2
        out_features = out_channels
        self.qlayer = QNN(
            in_features, out_features, nlayer=nlayer, qdev=qdev, emb=emb, meas=meas
        )

    def forward(self, x):
        B, C, H, W = x.shape  # [B, C, H, W]

        # padding: [B, C, H, W] -> [B, C, H+K-1, W+K-1]
        x = torch.nn.functional.pad(x, pad=self.pad)
        # unfolding: -> [B, CK^2, hw]
        x = torch.nn.functional.unfold(
            x, kernel_size=(self.kernel_size, self.kernel_size), stride=self.stride
        )
        # transposing: [B, CK^2, hw] -> [B, hw, CK^2]
        x = x.transpose(1, 2)

        # QNN
        x = self.qlayer(x)  # [B, hw, C'K^2]
        # anti-transposing: -> [B, C'K^2, hw]
        x = x.transpose(1, 2)
        # folding: -> [B, C', H/s, W/s]
        # x = torch.nn.functional.fold(x, output_size=(H, W), kernel_size=1, stride=self.stride)
        x = x.view(B, -1, (H - 1) // self.stride + 1, (W - 1) // self.stride + 1)

        return x


if __name__ == "__main__":
    # test use case
    def get_args():
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--kernel", "-k", default=1, type=int)
        parser.add_argument("--channel", "-c", default=[1, 1], type=int, nargs=2)
        parser.add_argument("--size", "-s", default=[4, 3], type=int, nargs=2)

        parser.add_argument("--model", default="quanv", type=str)
        parser.add_argument("--stride", "-S", default=1, type=int)
        parser.add_argument("--layer", default=2, type=int)
        parser.add_argument("--dev", default="default.qubit", type=str)
        parser.add_argument("--emb", default="amp", type=str)
        parser.add_argument("--meas", default="probs", type=str)

        parser.add_argument("--lr", default=0.05, type=float)
        parser.add_argument("--epoch", default=1000, type=int)
        parser.add_argument("--batch", "-b", default=2, type=int)

        parser.add_argument("--cuda", action="store_true")
        return parser.parse_args()

    args = get_args()
    print(args)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    if args.model == "qnn":
        model = QNN(
            in_features=args.size[1], out_features=2, nlayer=args.layer, qdev=args.dev
        )
    else:
        model = Quanv2d(
            in_channels=args.channel[1],
            out_channels=args.channel[0],
            kernel_size=args.kernel,
            stride=args.stride,
            nlayer=args.layer,
            qdev=args.dev,
        )
    model = model.to(device)

    x = torch.randn(
        [args.batch, args.channel[1], args.size[0], args.size[1]], device=device
    )
    print("x", x)
    y = model(x)
    print("y", y)
    print("in/out", x.shape, y.shape)

    for name, param in model.named_parameters():
        print(name, param.numel(), param.data)

    # train
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for e in range(args.epoch):
        model.train()
        model.zero_grad()

        input = torch.randn(
            [args.batch, args.channel[1], args.size[0], args.size[1]], device=device
        )
        # input = torch.ones(args.batch, args.channel[1], args.size[0], args.size[1])
        # print(input.shape)
        output = model(input)
        # print(output.shape)
        loss = torch.mean(output**2)

        loss.backward()
        opt.step()
        print(e, loss.item())
