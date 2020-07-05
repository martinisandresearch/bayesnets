#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import attr

from matplotlib import pyplot as plt, animation
import seaborn as sns
import numpy as np

from typing import List, Optional, Callable, Union, Iterable


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
    """Just for automcomplete, but this is the expected API"""
    artists = attr.ib(init=False, default=[])

    @property
    def num_frames(self) -> int:
        return 0

    def init(self, ax: plt.Axes):
        pass

    def animate(self, frame: int):
        pass


def validate_ax_kwargs(**kwargs):
    """
    This is an attempt to validate the kwargs on definition rather than at runtime for quicker
    feedback to the developer. Could also be used ahead of time, though it's not perfect

    Raises:
        ValueError: in the case of invalid arguments
    """
    cls = plt.Axes
    missing = []
    for k in kwargs:
        if not hasattr(cls, k):
            missing.append(k)
    if missing:
        raise ValueError("plt.Axes does not have {}".format(",".join(missing)))


def kwargs_hook(**kwargs):
    """
    Our animation library uses hooks to allow once of configuration of an axes. These
    include things like plotting the true y value or setting sensible x/y limits for the plot

    This can be time consuming to do, so this provides an easy way to do so, allowing
    one to call a

    Examples:
        >>> kwargs_hook(set_title="Goofy Experiment", set_ylabel="loss")

    Raises:
        ValueError: will attempt to validate the kwargs ahead of time

    """
    # get an early error if this is a problem
    # rather than at animate time
    validate_ax_kwargs(**kwargs)

    def inner_hook(ax):
        for k, v in kwargs.items():
            try:
                func = getattr(ax, k)
            except AttributeError:
                raise ValueError(f"Used a method {k}")
            except Exception as e:
                raise ValueError("Unexpected failure") from e
            else:
                func(v)

    return inner_hook


def apply_hook(ax: plt.Axes, hook: Union[Iterable[Callable], Callable, None]):
    if not hook:
        return
    try:
        for h in hook:
            h(ax)
    except TypeError:
        # can't iterate
        hook(ax)


@attr.s
class LineSwarm(SwarmPlot):
    x = attr.ib(type=np.ndarray)  # shape of (N,)
    data = attr.ib(type=np.ndarray)  # shape of (num_lines, timestep, N)
    hook = attr.ib(type=List[Callable], factory=list)

    @classmethod
    def standard(cls, xd, yd, data, **kwargs):
        validate_ax_kwargs(**kwargs)
        hook = [lambda ax: ax.plt(xd, yd, '.'), kwargs_hook(**kwargs)]
        return cls(xd, data, hook)

    @classmethod
    def auto_range(cls, xd, data, **kwargs):
        mx, mn = [data.max(), data.min()]

        def set_lim(ax: plt.Axes):
            ax.set_ylim(mn, mx)
            ax.set_xlim(xd.min(), xd.max())

        hook = [set_lim, kwargs_hook(**kwargs)]
        return cls(xd, data, hook)

    @property
    def num_frames(self) -> int:
        return self.data.shape[1]

    def init(self, ax: plt.Axes):
        self.artists = []
        apply_hook(ax, self.hook)
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
            # print(p.artists)
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
