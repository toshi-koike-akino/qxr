# train.py
import torch
import argparse
import prep_eden
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import plotly.express
import qnet
import PIL.Image


def get_args():
    parser = argparse.ArgumentParser(__name__)
    parser.add_argument(
        "--data",
        default="data/eden.npz",
        type=str,
        help="data npz file",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--sample", default=None, type=int)
    parser.add_argument("--batch", default=50, type=int)
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--patch", default=16, type=int)
    parser.add_argument("--kernel", default=3, type=int, help="CNN kernel size")
    parser.add_argument("--act", default="LeakyReLU", type=str)
    parser.add_argument(
        "--gen",
        default=[3, 10, 1],
        type=int,
        nargs="+",
        help="channel sizes for generator model",
    )
    parser.add_argument(
        "--disc",
        default=[4, 10, 20],
        type=int,
        nargs="+",
        help="channel sizes for discriminator model",
    )
    parser.add_argument(
        "--critic",
        default=1,
        type=int,
        help="number of critic iterations per generator update",
    )
    parser.add_argument(
        "--penalty", default=20, type=float, help="gradient penalty factor"
    )

    parser.add_argument("--dev", default="default.qubit", type=str)
    parser.add_argument("--emb", default="amp", type=str)
    parser.add_argument("--meas", default="probs", type=str)
    return parser.parse_args()


def plot_image(rgb, dep, vmax=15):
    print(dep.shape, rgb.shape)

    rgb = PIL.Image.fromarray(rgb, "RGB")
    rgb.show()
    rgb.save("rgb.png")

    dep = (np.tile(dep, (1, 1, 3)) / vmax * 255).astype("uint8")
    print(dep.shape, dep[200, 450])
    dep = PIL.Image.fromarray(dep, "RGB")
    dep.show()
    dep.save("dep.png")

    # plt.figure()
    # plt.imshow(dep, cmap="gray", vmin=0, vmax=vmax)
    # plt.savefig("dep.png")

    # plt.figure()
    # plt.imshow(rgb)
    # plt.savefig("rgb.png")
    # plt.show()
    # fig = plotly.express.imshow(dep)
    # fig.show()
    # fig = plotly.express.imshow(rgb)
    # fig.show()


def plot_histogram(deps, bins=100):
    hist = np.histogram(deps.ravel(), bins=bins)
    print(hist)
    plt.figure()
    plt.bar(hist[1][:-1], hist[0])
    plt.savefig("hist.png")


def get_data(file, cutoff=10):
    deps, rgbs = prep_eden.load_npz(file)

    # depth: background plane 0 -> cutoff
    deps[deps == 0] = cutoff
    deps[deps > cutoff] = cutoff

    plot_image(rgbs[0], deps[0], vmax=cutoff)
    plot_histogram(deps)

    # [B, H, W, C] -> [B, C, H, W]
    deps = np.moveaxis(deps, -1, 1)
    rgbs = np.moveaxis(rgbs, -1, 1)

    # np -> tensor
    deps = torch.FloatTensor(deps)  # * 0 + 0.01
    rgbs = torch.FloatTensor(rgbs)

    # tensor dataset
    dataset = torch.utils.data.TensorDataset(rgbs, deps)
    return dataset


def seeding(seed):
    if seed > 0:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# generator model: Hybrid classical-quantum CNN
def get_gen(args):
    model = [torch.nn.BatchNorm2d(args.gen[0])]
    for k in range(len(args.gen) - 1):
        model.append(
            torch.nn.Conv2d(
                args.gen[k],
                args.gen[k + 1],
                args.kernel,
                padding="same",
                padding_mode="reflect",
            )
        )
        model.append(getattr(torch.nn, args.act)())
        model.append(torch.nn.BatchNorm2d(args.gen[k + 1]))
    # QNN Conv layer
    model.append(
        qnet.Quanv2d(
            args.gen[-1], 1, args.kernel, qdev=args.dev, emb=args.emb, meas=args.meas
        )
    )
    model.append(torch.nn.BatchNorm2d(1))  # scale-up QNN output

    model = torch.nn.Sequential(*model)
    print(model)
    return model


# discriminator model
def get_disc(args):
    model = []
    model.append(torch.nn.BatchNorm2d(args.disc[0]))
    for k in range(len(args.disc) - 1):
        model.append(
            torch.nn.Conv2d(args.disc[k], args.disc[k + 1], args.kernel, stride=2)
        )
        model.append(getattr(torch.nn, args.act)())
        model.append(torch.nn.BatchNorm2d(args.disc[k + 1]))
    model.append(torch.nn.Flatten())
    model.append(torch.nn.LazyLinear(20))
    model.append(getattr(torch.nn, args.act)())
    model.append(torch.nn.LazyLinear(1))

    model = torch.nn.Sequential(*model)
    return model


# https://github.com/Lornatang/WassersteinGAN_GP-PyTorch/blob/master/wgangp_pytorch/utils.py
def calculate_gradient_penalty(model, real, fake, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones_like(model_interpolates, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty


def main():
    args = get_args()
    print(args)

    # random seed
    seeding(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset = get_data(args.data)
    sampler = torch.utils.data.RandomSampler(dataset, num_samples=args.sample)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, sampler=sampler
    )

    gen = get_gen(args).to(device)
    disc = get_disc(args).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), args.lr)
    disc_opt = torch.optim.Adam(disc.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        gen_opt, patience=args.patience, verbose=True
    )
    criterion = torch.nn.MSELoss()

    for epoch in range(args.epoch):
        total_loss = 0
        total_mse = 0
        for X, Y in loader:
            # update disc: max E D(x) - E D(G(x))
            for k in range(args.critic):
                # TODO: to use positional encoding of patch location
                Xreal, Yreal = gen_patch(X, Y, args.patch)
                Xreal, Yreal = Xreal.to(device), Yreal.to(device)

                disc.zero_grad()
                gen.zero_grad()
                real = torch.cat((Xreal, Yreal), 1)  # [B, 4, H, W]
                real_out = disc(real)

                Yfake = gen(Xreal)
                fake = torch.cat((Xreal, Yfake), 1)  # [B, 4, H, W]
                fake_out = disc(fake.detach())

                # gradient penalty: https://arxiv.org/pdf/1704.00028v3.pdf
                gp = calculate_gradient_penalty(disc, real, fake, device)

                disc_loss = fake_out.mean() - real_out.mean() + args.penalty * gp
                disc_opt.zero_grad()
                disc_loss.backward()
                disc_opt.step()
                print("critic", k, disc_loss.item(), gp.item())

            # update generator: min E D(G(x))
            Xreal, Yreal = gen_patch(X, Y, args.patch)
            Xreal, Yreal = Xreal.to(device), Yreal.to(device)

            disc.zero_grad()
            gen.zero_grad()
            Yfake = gen(Xreal)
            mse = criterion(Yreal, Yfake)  # just for log

            fake = torch.cat((Xreal, Yfake), 1)
            fake_out = disc(fake)
            disc_loss = -fake_out.mean()

            loss = disc_loss if args.critic else mse

            gen_opt.zero_grad()
            loss.backward()
            gen_opt.step()

            print(epoch, mse.item(), disc_loss.item())

            total_loss += loss.item()
            total_mse += mse.item()

        print(epoch, total_mse / len(loader), total_loss / len(loader))
        scheduler.step(total_loss)

    # save models
    fname = (
        "gen"
        + f"_p{args.patch}k{args.kernel}_c"
        + ("_".join(map(str, args.gen)))
        + ".pt"
    )
    save_model(gen, fname)
    save_model(disc, fname.replace("gen", "disc", 1))


def save_model(model, fname, root="models"):
    os.makedirs(root, exist_ok=True)
    fname = os.path.join(root, fname)
    print("saving", fname)
    torch.save(model, fname)


def gen_patch(X, Y, size):
    # joint RGB-D crop
    B, C, H, W = X.shape
    if H < size or W < size:
        raise ValueError(
            f"Required patch size {size} is larger than input image size {(H, W)}"
        )

    i = torch.randint(0, H - size + 1, size=(B,))
    j = torch.randint(0, W - size + 1, size=(B,))

    # X = X[:, :, i : i + size, j : j + size]
    # Y = Y[:, :, i : i + size, j : j + size]

    X2 = []
    Y2 = []
    for k in range(B):
        X2.append(X[k, :, i[k] : i[k] + size, j[k] : j[k] + size])
        Y2.append(Y[k, :, i[k] : i[k] + size, j[k] : j[k] + size])

    return torch.stack(X2), torch.stack(Y2)


if __name__ == "__main__":
    main()
