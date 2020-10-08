import copy
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import core_causal_features

class Agent(object):
    """
    Abstract interface for Agents.
    """
    def behave(self, I=None, H=None):
        """
        Return the (binary) behavior of the agent, given that 
        an image I is presented and non-visual variables are
        in state H.

        Parameters:
        -----------
        I : numpy array 
            A binary image or images.
        H : numpy array or None
            A (possibly non-binary) "hidden variable". Should either
            be None or have the same number of entries as I. If H is 
            given, the agent should respond according to the observa-
            tional model in nature with given H. If H is None, the a-
            gent should simulate the experimental setting, in which 
            H is marginalized out.

        Returns:
        --------
        T : numpy array
            An array of binary behavior values.
        """
def _remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def _framed_pic(ax, img):
    """
    Display a black-and-white image in a tranparent gray frame.
    """
    if len(img.shape)==1 or img.shape[1]==1:
        ax.imshow(img.reshape((np.sqrt(img.size), np.sqrt(img.size))), 
                cmap='Greys', interpolation='nearest', vmin=0, vmax=1)
    else:
        ax.imshow(img, cmap='Greys', interpolation='nearest', 
                  vmin=0, vmax=1)
    ax.add_patch(Rectangle((0,0), 1, 1, transform=ax.transAxes, 
                    fc='none', lw=2, edgecolor='black', alpha=0.2))

def _plot_column(figH, x0, start_xs, end_xs, 
                 im_shape, txt, clr1, clr2=None, freqs=None):
    """
    Plot a column of original and manipulated images.
    """
    # Color pallette 
    ALPHA_solid=1.
    ALPHA_trans=0.2
    BLACK = (0, 0, 0)
    # Define the dimensions of the column.

    W_im = 0.09 # Image side len
    H_major = 0.1 # Total height of two-image row
    x = (0.25-2*W_im)/3 # Horizontal margin
    y = (H_major-W_im)/2 # Vertical margin
    W = 2*W_im+x # Width of two-image column    
    H_minor = H_major-2*y # Two-image row heigth without margins
    assert H_minor==W_im, 'Wrong dimensions!'
    H_title = 0.666*H_minor
    H_bar = H_minor-H_title-2*y

    ax_title=figH.add_axes([x0+x, (1-H_major)+2*y+H_bar, W, H_title])
    plt.axis('off')
    ax_title.add_patch(Rectangle((0, 0), 1, 1, 
                transform=ax_title.transAxes, fc=clr1, edgecolor='black',
                                 alpha=ALPHA_solid))
    ax_title.text(0.5, 0.4, txt, fontsize=20, ha='center', va='center')
    
    if freqs is not None:
        ax_bar=figH.add_axes([x0+x, (1-H_major)+y, 2*W+2*x, H_bar])
        ax_bar.yaxis.set_visible(False)
        ax_bar.set_xlim(0,1)
        ax_bar.set_xticks(np.linspace(0,1,6))
        ax_bar.set_xticklabels([])
        ax_bar.add_patch(Rectangle((0,0), freqs[0], 1,
                transform=ax_bar.transAxes, fc=clr1, edgecolor='none',
                                 alpha=ALPHA_solid))
        ax_bar.add_patch(Rectangle((freqs[0],0), 1-freqs[0], 1,
                transform=ax_bar.transAxes, fc=clr2, edgecolor='none',
                                 alpha=ALPHA_solid))
        
    for row_id in range(start_xs.shape[0]):
        # Plot the original image.
        ax_img = figH.add_axes([x0+x, (1-H_major*(row_id+2))+y, 
                                W_im, H_minor])
        plt.axis('off')
        _framed_pic(ax_img, start_xs[row_id])
        
        # Plot the manipulated image.
        ax_img = figH.add_axes([x0+x+W_im+x, (1-H_major*(row_id+2))+y, 
                                W_im, H_minor])
        plt.axis('off')
        _framed_pic(ax_img, end_xs[row_id])

def incorporate_queries(manip_pool, agent, data_train, data_valid):
    """ Update training and test sets using a manipulation pool
    and simulated agent's responses to manipulation queries.
    """
    # Put it all together and save.
    new_xy = [(manip[1][0], agent.behave(manip[1][0])) for manip in manip_pool.items()]
    new_X, new_y = zip(*new_xy)
    manip_err = 1-np.sum([(manip_pool.items()[i][1][1]>1./data_train.y_labels)*
        (manip_pool.items()[i][1][2]==new_y[i]) for i in range(len(new_y))])/float(len(new_y))
    print('Manipulation error: {}'.format(manip_err))
    (new_train, new_valid) = core_causal_features.update_dataset(np.asarray(new_X), np.atleast_2d(new_y).T, data_valid, data_train)
    return (new_train, new_valid)
