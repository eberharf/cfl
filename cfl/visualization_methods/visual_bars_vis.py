import random

import matplotlib.pyplot as plt
import numpy as np

from cfl.util.x_lbl_util import rows_where_each_x_class_occurs


def viewImagesAndLabels(images, n_rows, x_lbls):
    """
    shows images in matplotlib with labels displayed at the top of each image.
    Best for viewing a lot of images at once. Designed with visual bars images
    in mind.

    Parameters:
    images (3D np array): Array of images (must be aligned with x_lbls)
    n_rows (int): Number of rows of images to display.
    x_lbls (1D np array): labels to show at the top of each image.
                        Should be aligned with the images input
    """

    assert images.shape[0] == x_lbls.shape[0], "The number of images and \
    x_lbls should be equal"

    ##### TODO: FIRST: UNCOMMENT ALL THIS
    # # size of each image in pixels
    # FIG_SIZE = (10, 10)

    # # set the number of columns to display images over
    # # each column corresponds to a class
    # N_COLS = len(np.unique(x_lbls))

    # # find where each class label occurs
    # x_lbl_indices = rows_where_each_x_class_occurs(x_lbls)

    # # number of rows = minimum bt input n_rows and number of images in smallest class
    # # (so that we don't try to display more images than we possibly can)
    # n_rows_actual = min(n_rows, min([len(x) for x in x_lbl_indices]))

    # # create subplot for each image
    # fig, ax = plt.subplots(nrows=n_rows_actual, ncols=N_COLS, squeeze=False, figsize=FIG_SIZE)

    # ####TODO: THIS IS WHERE THE CODE STILL NEEDS TO BE WRITTEN
    # ## fill in the plots such that each column corresponds to a particular class
    # ## and each row contains one image from each class, with the label shown
    # ## above it
    # ## (similar to the way the output for this function used to look, but now sorted by class)







    #     #use the label for the image as the title of the subplot
    #     current_class = x_lbls[i]
    #     ax[row][col].title.set_text(str(int(current_class)))

    # #turn off axes for all the subplots (looks bad for images)
    # #This is the command to do that:  ax[row][col].axis("off")

    # fig.tight_layout()
    # plt.show()


def viewSingleImage(image_array, random_state=None):
    """chooses a random image from the image_array and displays it. Setting random state causes it to be reproducible"""
    ###TODO: write this code
    #choose a random image

    # show the image in matplotlib
    pass