#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import attr

from matplotlib import pyplot as plt, animation
import seaborn as sns
import numpy as np

from typing import List, Optional


def make_animation(xd, yd, data, title: str, destfile: str):
    """
    Convenience method to produce an animation - requires a refactor too

    Args:
        xd: np.ndarray (N,)
        yd: np.ndarray (N,)
        data: np.ndarray (Bees, epochs, N)
        title :str
        destfile: str
    """
    plt.rcParams["figure.figsize"] = (14.0, 7.0)
    sns.set()
    # Determine a proper format
    nepoch = data[0].shape[0]
    fig = plt.figure()
    ax = plt.axes()
    ax.set_title(title)
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


@attr.s
class SwarmPlot:
    """Just for automcomplete"""
    artists = attr.ib(init=False, default=[])

    @property
    def num_frames(self) -> int:
        return 0

    def init(self, ax: plt.Axes):
        pass

    def animate(self, frame: int):
        pass


@attr.s
class LineSwarm(SwarmPlot):
    x = attr.ib(type=np.ndarray)  # shape of (N,)
    data = attr.ib(type=np.ndarray)  # shape of (num_lines, timestep, N)
    hook = attr.ib(default=lambda ax: None)


    @property
    def num_frames(self) -> int:
        return self.data.shape[1]
    
    def init(self, ax: plt.Axes):
        self.artists = []
        self.hook(ax)
        for i in range(self.data.shape[0]):
            (liner,) = ax.plot([], [], lw=2)
            self.artists.append(liner)

    def animate(self, frame: int):
        for bee, line in enumerate(self.artists):
            line.set_data(self.x, self.data[bee][frame])


def make_init_func(plots: List[SwarmPlot], axes: List[plt.Axes]):
    """create a chained init for each frame"""

    def inner():
        all_artists = []
        for p, ax in zip(plots, axes):
            print(p.artists)
            p.init(ax)
            all_artists.extend(p.artists)
        return all_artists

    return inner


def make_animate_func(plots: List[SwarmPlot]):
    def inner(frame: int):
        all_artists = []
        for p in plots:
            p.animate(frame)
            all_artists.extend(p.artists)
        return all_artists
    return inner


def swarm_animate(plots: List[SwarmPlot], destfile: str, num_frames: Optional[int] = None):
    if len(plots) != 2:
        raise ValueError("Must be 2 plots while under development")

    if not num_frames:
        all_frames = {p.num_frames for p in plots}
        assert len(all_frames) == 1
        num_frames = all_frames.pop()

    plt.rcParams["figure.figsize"] = (14.0, 7.0)
    sns.set()
    fig = plt.figure()
    fig.subplots(2, 1)

    anim = animation.FuncAnimation(
        fig, make_animate_func(plots), init_func=make_init_func(plots, fig.axes),
        frames=num_frames, interval=20,
        blit=True, repeat=False
    )

    if not destfile.endswith(".mp4"):
        destfile = f"{destfile}.mp4"
    anim.save(destfile, fps=30, extra_args=["-vcodec", "libx264"])
    plt.close()
    print(f"Saved to {destfile}")
