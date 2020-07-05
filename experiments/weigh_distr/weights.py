#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import torch
from torch import nn

from swarm import core, animator, networks, util


@util.time_me
def make_hist_animation(hist_data, name):
    """
    Convenience method to produce an animation - requires a refactor too
    """
    import seaborn as sns
    import numpy as np

    # Determine a proper format
    nepoch = hist_data[0].shape[0]
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.path as path
    import matplotlib.animation as animation

    plt.rcParams["figure.figsize"] = (14.0, 7.0)
    sns.set()

    fig, ax = plt.subplots()
    plt.title(f"Histogram distributions of {name}")

    # histogram our data with numpy
    # comes in bee, epoch, value. We transpose and flatten last two
    beel, epochl, vall = hist_data.shape
    hist_data = hist_data.transpose(1, 0, 2)
    hist_data = hist_data.reshape(epochl, beel * vall)

    init_data = hist_data.flatten()
    n, bins = np.histogram(init_data, 100)

    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n / (epochl * 0.9)
    nrects = len(left)

    # here comes the tricky part -- we have to set up the vertex and path
    # codes arrays using moveto, lineto and closepoly

    # for each rect: 1 for the MOVETO, 3 for the LINETO, 1 for the
    # CLOSEPOLY; the vert for the closepoly is ignored but we still need
    # it to keep the codes aligned with the vertices
    nverts = nrects * (1 + 3 + 1)
    verts = np.zeros((nverts, 2))
    codes = np.ones(nverts, int) * path.Path.LINETO
    codes[0::5] = path.Path.MOVETO
    codes[4::5] = path.Path.CLOSEPOLY
    verts[0::5, 0] = left
    verts[0::5, 1] = bottom
    verts[1::5, 0] = left
    verts[1::5, 1] = top
    verts[2::5, 0] = right
    verts[2::5, 1] = top
    verts[3::5, 0] = right
    verts[3::5, 1] = bottom

    barpath = path.Path(verts, codes)
    patch = patches.PathPatch(barpath, facecolor="green", edgecolor="yellow", alpha=0.5)
    ax.add_patch(patch)

    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())

    def animate(i):
        # simulate new data coming in
        data = hist_data[i]
        # print(i, hist_data.shape, data)

        n, bins = np.histogram(data, 100)
        top = bottom + n
        verts[1::5, 1] = top
        verts[2::5, 1] = top

    ani = animation.FuncAnimation(fig, animate, nepoch, repeat=False)

    destfile = f"{name}.mp4"
    ani.save(destfile, fps=30, extra_args=["-vcodec", "libx264"])
    plt.close()


def get_firstwb(net):
    "Version that assumes we have a single Linear layer"
    for layer in net:
        if isinstance(layer, torch.nn.Linear):
            weight, bias = layer.parameters()
            return weight.flatten().detach().numpy().copy(), bias.detach().numpy().copy()


def bee_trainer(xt, yt, width=2, num_epochs=200):
    net = networks.flat_net(1, width, activation=nn.ReLU)

    optimiser = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    loss_func = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        optimiser.zero_grad()
        ypred = net(xt)

        loss = loss_func(ypred, yt)
        if torch.isnan(loss):
            raise RuntimeError("NaN loss, poorly configured experiment")

        loss.backward()
        optimiser.step()

        weight, bias, *_ = net.parameters()
        yield ypred, weight.detach().flatten().numpy().copy(), bias.detach().numpy().copy()


def main():
    import numpy as np

    xt = torch.linspace(-3 * np.pi, 3 * np.pi, 101)
    yt = torch.sin(xt)

    bp = {"xt": xt, "yt": yt, "width": 20, "num_epochs": 400}
    # bs = list(bee_trainer(**bp))
    res = core.swarm_train(bee_trainer, bp, num_bees=500, fields="ypred,weights,biases", seed=20)
    # from pprint import pprint
    # pprint(bs)
    # print(res["weights"].shape)
    # print(res["biases"])
    # print(res["biases"].max(), res["biases"].min())
    # make_hist_animation(res["biases"], "biases")
    bw = res["biases"] / res["weights"]
    print(bw.min(), bw.max())
    print(np.percentile(bw, [1, 5, 90, 95]))
    # print(bw)
    bw = bw.clip(-20, 20)

    ls = animator.LineSwarm.standard(xt.detach().numpy(), yt.detach().numpy(), res["ypred"][::10])
    hist = animator.HistogramSwarm.from_swarm(
        bw, 100, set_title="Biases/Weights", set_ylabel="Count"
    )
    animator.swarm_animate([ls, hist], "weight_distr.mp4")
    animator.swarm_animate([hist], "weights.mp4")


if __name__ == "__main__":
    main()
