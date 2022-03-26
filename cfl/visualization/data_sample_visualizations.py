'''
Two functions are for viewing images
'''

import random
import matplotlib.pyplot as plt
import numpy as np
from cfl.util.find_xlbl_locations import rows_where_each_x_class_occurs


def view_class_examples(images, im_shape, n_examples, x_lbls):
    """
    Shows images in matplotlib with labels displayed at the top of each image.
    Best for viewing a lot of images at once.

    Arguments:
        images (2D or 3D np array): Array of images (must be aligned with x_lbls)
            If 2D, axis 0 = samples, axis 1 = flattened image pixels
            If 3D, axis 0 = samples, axis 1 = image rows, axis 2 = image cols
        n_rows (int): Number of rows of images to display.
        x_lbls (1D np array): labels to show at the top of each image.
            Should be aligned with the images input
    Returns: None
    """

    assert images.shape[0] == x_lbls.shape[0], "The number of images and \
    x_lbls should be equal"

    if images.ndim == 2:  # reshape images
        images = np.reshape(
            images, (images.shape[0], im_shape[0], im_shape[1]))
    elif images.ndim == 3:  # images already 2D
        pass
    else:
        raise ValueError('Expected images to be 2D or 3D. ' +
                         'Instead, images.ndim={}'.format(images.ndim))

    # set the number of columns to display images over
    # each column corresponds to a class
    N_COLS = len(np.unique(x_lbls))

    # find where each class label occurs
    x_lbl_indices = rows_where_each_x_class_occurs(x_lbls)

    # number of rows = minimum bt input n_rows and number of images in smallest class
    # (so that we don't try to display more images than we possibly can)
    N_ROWS = min(n_examples, min([len(x) for x in x_lbl_indices]))
    if N_ROWS == 0:
        print('You have at least one class with no examples. Currently, this' +
              'function limits the number of examples displayed by the size of the smallest class')
        return

    # create subplot for each image
    fig, ax = plt.subplots(nrows=N_ROWS, ncols=N_COLS,
                           figsize=(N_COLS*2, N_ROWS*2))
    for i in range(N_COLS):
        idx_to_plot = np.random.choice(x_lbl_indices[i], N_ROWS, replace=False)
        for j in range(N_ROWS):
            ax[j, i].imshow(images[idx_to_plot[j]])
            ax[j, i].set_title('Class {}'.format(i))
            ax[j, i].axis('off')
    fig.tight_layout()
    plt.show()


def view_random_example(image_array, random_state=None):
    """
    Chooses a random image from the image_array and displays
    it. Setting random state causes it to be reproducible.
    Arguments:
        image_array (np.ndarray) : array of images of size 
            (n_samples, image_dim1, img_dim2)
        random_state (int) : random state to set rng.choice to for selecting
            random images to display. Default is None.
    Returns: None
    """
    rng = np.random.default_rng(seed=random_state)
    # choose random img idx
    idx = rng.choice(image_array.shape[0], replace=False)

    fig, ax = plt.subplots()
    ax.imshow(image_array[idx])
    ax.axis('off')
    plt.show()
