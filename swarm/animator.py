#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

from matplotlib import pyplot as plt, animation
import seaborn as sns

from swarm import util


@util.time_me
def make_animation(xd, yd, data, title, destfile):
    """
    Convenience method to produce an animation - requires a refactor too
    """
    plt.rcParams["figure.figsize"] = (14.0, 7.0)
    sns.set()
    # Determine a proper format
    nepoch = data[0].shape[0]
    fig = plt.figure()
    ax = plt.axes()
    plt.title(title)
    ax.plot(xd, yd, ".")

    line_ref = []
    for i in range(len(data)):
        (liner,) = ax.plot([], [], lw=2)
        line_ref.append(liner)

    def init():
        for line in line_ref:
            line.set_data([], [])
        return line_ref

    def animate(i):
        for dnum, line in enumerate(line_ref):
            line.set_data(xd, data[dnum][i])
        return line_ref

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=nepoch, interval=20, blit=True, repeat=False
    )
    # i think interval is basically overwritten by fps
    # increase fps if you feel like the animation is moving too slowly
    if destfile:
        # secret skip
        anim.save(destfile, fps=30, extra_args=["-vcodec", "libx264"])
    plt.close()
