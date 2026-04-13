"""Display utilities for canora — specshow, waveshow, cmap.

Pure Python module wrapping matplotlib. Mirrors librosa.display API.
"""

import warnings

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.axes as mplaxes
    import matplotlib.ticker as mplticker
    from matplotlib.colors import LinearSegmentedColormap

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for canora.display. "
            "Install it with: pip install matplotlib"
        )


# ============================================================
# Axis formatters
# ============================================================

if HAS_MATPLOTLIB:

    class TimeFormatter(mplticker.Formatter):
        """Format time values in seconds to human-readable strings."""

        def __init__(self, lag=False):
            self.lag = lag

        def __call__(self, x, pos=None):
            sign = ""
            if x < 0:
                sign = "-"
                x = -x
            if x >= 3600:
                s = f"{sign}{int(x // 3600)}:{int((x % 3600) // 60):02d}:{x % 60:05.2f}"
            elif x >= 60:
                s = f"{sign}{int(x // 60)}:{x % 60:05.2f}"
            else:
                s = f"{sign}{x:.2f}"
            return s

    class NoteFormatter(mplticker.Formatter):
        """Format Hz values as note names."""

        def __init__(self, octave=True, major=True):
            self.octave = octave
            self.major = major

        def __call__(self, x, pos=None):
            if x <= 0:
                return ""
            try:
                from canora._canora import hz_to_note
                return hz_to_note(float(x))
            except Exception:
                return f"{x:.0f}"

    class ChromaFormatter(mplticker.Formatter):
        """Format chroma bin indices as note names."""

        NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        def __call__(self, x, pos=None):
            idx = int(round(x)) % 12
            return self.NOTE_NAMES[idx]

    class TonnetzFormatter(mplticker.Formatter):
        """Format tonnetz dimension indices."""

        LABELS = ["5th x", "5th y", "m3rd x", "m3rd y", "M3rd x", "M3rd y"]

        def __call__(self, x, pos=None):
            idx = int(round(x))
            if 0 <= idx < len(self.LABELS):
                return self.LABELS[idx]
            return ""


# ============================================================
# Colormap
# ============================================================


def cmap(data=None, robust=True, cmap_seq="magma", cmap_div="coolwarm"):
    """Select a colormap based on data characteristics.

    Parameters
    ----------
    data : array-like, optional
        If provided, choose diverging cmap if data has both positive and negative values.
    robust : bool
        If True, use percentile-based limits.
    cmap_seq : str
        Sequential colormap name.
    cmap_div : str
        Diverging colormap name.

    Returns
    -------
    matplotlib.colors.Colormap
    """
    _check_matplotlib()
    if data is not None:
        data = np.asarray(data)
        if np.any(data < 0) and np.any(data > 0):
            return plt.get_cmap(cmap_div)
    return plt.get_cmap(cmap_seq)


# ============================================================
# specshow
# ============================================================


def specshow(
    data,
    *,
    x_coords=None,
    y_coords=None,
    x_axis=None,
    y_axis=None,
    sr=22050,
    hop_length=512,
    n_fft=None,
    fmin=None,
    fmax=None,
    ax=None,
    **kwargs,
):
    """Display a spectrogram/chromagram/feature matrix.

    Parameters
    ----------
    data : array-like, shape (d, n)
        Matrix to display.
    x_axis : str or None
        Type for x-axis: 'time', 'frames', 's', 'ms', None.
    y_axis : str or None
        Type for y-axis: 'linear', 'log', 'mel', 'hz', 'chroma', 'tonnetz', None.
    sr : int
        Sample rate.
    hop_length : int
        Hop length for time axis.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    **kwargs
        Passed to ax.pcolormesh().

    Returns
    -------
    matplotlib.collections.QuadMesh
    """
    _check_matplotlib()

    data = np.asarray(data)

    if ax is None:
        ax = plt.gca()

    # Default colormap
    if "cmap" not in kwargs:
        kwargs["cmap"] = cmap(data)

    # Build coordinate arrays
    n_y, n_x = data.shape

    if x_coords is None:
        if x_axis in ("time", "s"):
            x_coords = np.arange(n_x + 1) * hop_length / sr
        elif x_axis == "ms":
            x_coords = np.arange(n_x + 1) * hop_length / sr * 1000
        elif x_axis == "frames":
            x_coords = np.arange(n_x + 1)
        else:
            x_coords = np.arange(n_x + 1)

    if y_coords is None:
        if y_axis in ("linear", "hz", "log"):
            if n_fft is not None:
                y_coords = np.linspace(0, sr / 2, n_y + 1)
            else:
                y_coords = np.arange(n_y + 1)
        elif y_axis == "mel":
            y_coords = np.arange(n_y + 1)
        elif y_axis == "chroma":
            y_coords = np.arange(n_y + 1)
        elif y_axis == "tonnetz":
            y_coords = np.arange(n_y + 1)
        else:
            y_coords = np.arange(n_y + 1)

    # Plot
    img = ax.pcolormesh(x_coords, y_coords, data, shading="flat", **kwargs)

    # Decorate axes
    if x_axis == "time" or x_axis == "s":
        ax.xaxis.set_major_formatter(TimeFormatter())
        ax.set_xlabel("Time (s)")
    elif x_axis == "frames":
        ax.set_xlabel("Frames")

    if y_axis == "log":
        ax.set_yscale("log")
        ax.set_ylabel("Hz")
    elif y_axis in ("linear", "hz"):
        ax.set_ylabel("Hz")
    elif y_axis == "mel":
        ax.set_ylabel("Mel")
    elif y_axis == "chroma":
        ax.yaxis.set_major_formatter(ChromaFormatter())
        ax.set_ylabel("Pitch class")
    elif y_axis == "tonnetz":
        ax.yaxis.set_major_formatter(TonnetzFormatter())
        ax.set_ylabel("Tonnetz")

    return img


# ============================================================
# waveshow
# ============================================================


def waveshow(y, *, sr=22050, ax=None, **kwargs):
    """Display a waveform.

    Parameters
    ----------
    y : array-like, shape (n,)
        Audio signal.
    sr : int
        Sample rate.
    ax : matplotlib.axes.Axes, optional
    **kwargs
        Passed to ax.plot().

    Returns
    -------
    list of matplotlib.lines.Line2D
    """
    _check_matplotlib()

    y = np.asarray(y)
    if ax is None:
        ax = plt.gca()

    times = np.arange(len(y)) / sr

    kwargs.setdefault("color", "steelblue")
    kwargs.setdefault("linewidth", 0.5)

    lines = ax.plot(times, y, **kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.xaxis.set_major_formatter(TimeFormatter())

    return lines
