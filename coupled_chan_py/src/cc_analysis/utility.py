from dataclasses import dataclass
from typing import Any
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path
from typing import Optional

class Config:
    _instance: Optional["Config"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, data_path: Optional[str] = None, save_path: Optional[str] = None):
        plt.rcParams.update({'font.size': 18})

        if data_path is not None:
            self.DATA_PATH = Path(data_path).expanduser().resolve()
        if save_path is not None:
            self.SAVE_PATH = Path(save_path).expanduser().resolve()

    def __repr__(self) -> str:
        return f"Config(DATA_PATH={getattr(self, 'DATA_PATH', None)}, SAVE_PATH={getattr(self, 'SAVE_PATH', None)})"

CFG = Config("../data", "../plots")

def __inherit_doc(from_func):
    def decorator(func):
        func.__doc__ = (func.__doc__ or "") + "\n\n" + (from_func.__doc__ or "")
        return func
    return decorator


@__inherit_doc(plt.subplots)
def plot(**kwargs) -> tuple[Figure, Axes]:
    """Wrapper around single plot for `plt.subplots` with defaults (grid + ticks)."""
    fig, ax = plt.subplots(**kwargs)
    ax.grid()
    ax.tick_params(which='both', direction="in")

    return fig, ax

@dataclass
class AxesArray:
    array: Any
    nrows: int
    ncols: int
    
    def __getitem__(self, key) -> Axes:
        return self.array[key] if self.nrows * self.ncols > 1 else self.array
    
    def __iter__(self):
        return AxesIter(self, 0)
    
    def transposed(self):
        return AxesArray(self.array.T, self.ncols, self.nrows)
    
@dataclass
class AxesIter:
    axes: AxesArray
    current: int
    row_major: bool = True

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.axes.ncols * self.axes.nrows:
            raise StopIteration
        else:
            if self.axes.ncols == 1 or self.axes.nrows == 1:
                self.current += 1

                return self.axes[self.current - 1]
            else:
                j = self.current % self.axes.ncols
                i = self.current // self.axes.ncols

                self.current += 1
                return self.axes[i, j] if self.row_major else self.axes[j, i]

@__inherit_doc(plt.subplots)
def plot_many(nrows: int, ncols: int, **kwargs) -> tuple[Figure, AxesArray]:
    """Wrapper around multiple plots for `plt.subplots` with defaults (grid + ticks)."""
    fig, axes = plt.subplots(nrows, ncols, **kwargs)

    axes_array = AxesArray(axes, nrows, ncols)
    for ax in axes_array:
        ax.grid()
        ax.tick_params(which='both', direction="in")

    return fig, axes_array