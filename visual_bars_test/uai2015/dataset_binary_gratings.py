import numpy as np
from pylearn2.datasets import dense_design_matrix

class GRATINGS(dense_design_matrix.DenseDesignMatrix):
    """ A binary image dataset created from the following probabilities:
    [Below, H is a "hidden variable", VG is "vertical grating" 
     and HG stands for "horizontal grating".]

    P(H=0) = 0.5, P(H=1) = 0.5
    P(# of VGs added to image \in {2,4} | H=0) = 1
    P(# of VGs added to image \in {1,3} | H=1) = 0
    P(# of HGs added to image \in {2,4) = 0.5
    P(# of HGs added to image \in {1,3) = 0.5
    """

    def __init__(self, agent, n_samples=1000, 
                 im_shape=(10, 10), noise_lvl=0):
        self.args = locals()
        # the images in a topological arrangement; start with 
        # low-density, random noise images.
        X_topo = np.random.rand(n_samples, im_shape[0], 
                                im_shape[1]) < noise_lvl
        X_topo = X_topo.astype('float32')
        
        # possible values for the number of HGs and VGs
        HG_val_poss = [0] #[2, 4]
        VG_val_poss = [0] #[2, 4]

        # Select hidden variable values.
        Hs = np.random.rand(n_samples)<0.5

        # Select numbers of VGs and HGs.
        VGs = np.random.choice(VG_val_poss, n_samples)+Hs
        HGs = np.random.choice(HG_val_poss, n_samples)+\
                  (np.random.rand(n_samples)<0.5)

        # Make images with randomly placed Gs.
        for sample_id in range(n_samples):
            # Place the vertical bars.
            VG_locs = np.random.choice(range(im_shape[1]), 
                                       VGs[sample_id], replace=False)
            HG_locs = np.random.choice(range(im_shape[0]), 
                                       HGs[sample_id], replace=False)
            X_topo[sample_id, HG_locs, :] = 1.
            X_topo[sample_id, :, VG_locs] = 1.

        y = np.atleast_2d([agent.behave(X_topo[i], Hs[i]) for i in
                           range(n_samples)]).T
        m, r, c = X_topo.shape
        X_topo = X_topo.reshape(m, r, c, 1)

        # Initiate the dense array matrix.
        super(GRATINGS, self).__init__(topo_view=X_topo, 
                                       y=y, y_labels=np.unique(y).size)
        
