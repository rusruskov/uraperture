import numpy as np
import scipy.fftpack as sfft
import matplotlib.pyplot as plt
from PIL import Image

def image_to_array(path):
    """
    A function which inputs an image and returns an array of the pixels.

    Parameters
    ----------
    path : string

    Returns
    -------
    arr - numpy array
    """

    #Opens the image.
    image = Image.open(path)
    #Converts to greyscale.
    image = image.convert('1')
    #Establishes the size of the array.
    X = image.size[0]
    Y = image.size[1]

    #Creates an array where each value is the value of the
    #pixel of the image.
    arr = [[image.getpixel((x,y)) for x in range(X)] for y in range(Y)]
    #Converts to numpy array.
    arr = np.array(arr)

    return arr/255


def compute_intensity(aperture):
    """
    Computes the screen diffraction pattern.

    Parameters
    ----------
    aperture : 2D numpy array

    Returns
    -------
    screen : a 2D numpy array
    """

    #Computes a 2D fast fourier transform of the aperture
    screen = sfft.fft2(aperture)
    #Scipy magic
    screen = np.abs(sfft.fftshift(screen))

    return screen

def crop(screen, k):
    """
    Crops the screen (or any numpy array) around its centre leaving k-percent
    of the original array.

    Parameters
    ----------
    screen : numpy array
        the array needed to be cropped

    k : float
        the percentage by which the array is cropped
    """

    x = screen.shape[0]
    y = screen.shape[1]
    cropx = int(k*x)
    cropy = int(k*y)
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    screen = screen[starty:starty+cropy, startx:startx+cropx]/screen.max()

    return screen


def plot(aperture, screen, figsize=None, title=None, titlesize=None):
    """
    A simple plotting function, designed to facillitate the plotting of results.

    Parameters
    ----------
    aperture : 2D numpy array
        the aperture array to be plotted

    screen : 2D numpy array
        the screen array to be plotted

    figsize : tuple; optional
        size of the figure in inches

    title : string; optional
        title of the figure

    titlesize : int; optional
        font size for the title

    Returns
    -------
    fig : matplotlib.pyplot.figure instance
    """

    #Presets figure
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, size=titlesize)

    #Getting axes object for first plot
    ax1 = plt.subplot(121)
    #Displays image
    plt.imshow(aperture)
    plt.title('Aperture')
    #Removes pixel numbers on axes
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    #Removes axes ticks
    plt.xticks([])
    plt.yticks([])

    #Analogously
    ax2 = plt.subplot(122)
    plt.imshow(screen)
    plt.title('Screen')
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    plt.xticks([])
    plt.yticks([])

    return fig
