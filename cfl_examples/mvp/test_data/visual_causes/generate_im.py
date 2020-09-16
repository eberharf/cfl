#Jenna Kahn
#adapted from dataset_binary_gratings.py

import numpy as np
import math
import matplotlib.pyplot as plt

"""
    A binary image dataset created from the following probabilities:
[Below, H is a "hidden variable", VB is "vertical bars" 
    and HB stands for "horizontal bars".]
H causes hidden vertical bars and increases the probability of T (the target
behavior). Horizontal bars also increase the probability of T directly, 
but vertical bars do not. 


P(H=0) = 0.5, P(H=1) = 0.5


Ground truth: 
C = the presence of horizontal bars in image 
H = presence of hidden causal variable

class labels and P(T) for each class:
0. p(T|C=0,H=0) = 0.1
1. p(T|C=0,H=1) = 0.4
2. P(T|C=1,H=0) = 0.7
3. P(T|C=1,H=1) = 1.
"""

class VisualBarsData(): 

    def __init__(self, n_samples=1000, im_shape=(10,10), noise_lvl=0):
        self.n_samples = n_samples
        self.X_topo, self.Hs, self.HBs = self.generate_images(n_samples, im_shape, noise_lvl)
        self.gt_labels = self.ground_truth_classes()
        self.target_vals = self.generate_target()

    def __repr__(self): 
        return str(self.X_topo)

    def generate_images(self, n_samples, im_shape, noise_lvl):
        '''
        generates binary images containing some combination of vertical 
        bars and/or horizontal bars (or neither)

        inputs: 
        n_samples = number of images to generate 
        im_shape = dimensions of each image 
        noise_lvl = amount of noise in images (can be between 0 and 1)
        '''

        # X_topo = array containing each image (each val in array represents a pixel)
        X_topo = np.random.rand(n_samples, im_shape[0],  
                                im_shape[1]) < noise_lvl
        X_topo = X_topo.astype('float32')
        
        # possible values for the number of HBs and VBs
        # starting with 0 will allow for up to 1 HB and 1 VB in each image
        HB_val_poss = [0]
        VB_val_poss = [0] 

        # Select hidden variable values.
        Hs = np.random.rand(n_samples)<0.5 #Hs = array containing presence/absence of hidden var for each image 

        # Select numbers of VBs and HBs.
        VBs = np.random.choice(VB_val_poss, n_samples)+Hs
        HBs = np.random.choice(HB_val_poss, n_samples)+\
                    (np.random.rand(n_samples)<0.5)

        # Make images with randomly placed Gs.
        for sample_id in range(n_samples):
            # Place the vertical bars.
            VB_locs = np.random.choice(range(im_shape[1]), 
                                        VBs[sample_id], replace=False)
            HB_locs = np.random.choice(range(im_shape[0]), 
                                        HBs[sample_id], replace=False)
            X_topo[sample_id, HB_locs, :] = 1.
            X_topo[sample_id, :, VB_locs] = 1.

        return X_topo, Hs, HBs


    def ground_truth_classes(self):
        """ 
        modified from behave() in ai_gratings.py 

        Input 
        X_topo - array of binary images with some combo of horiz/vert bars 
        Hs - array, aligned with X_topo, saying whether hidden var is active for each image
        HBs - array, aligned with X_topo, saying whether each image contains a horiz bar or not 
        """

        # THR = 0.9    #TBH I don't know what these are about 
        # C = int(np.sum(I.reshape(self.visual_reshape).sum(axis=1) >= np.floor(THR*self.visual_reshape[0]))>0)

        gt_labels = np.zeros(self.n_samples) #gt_labels = array containing "ground truth" class labels for each image in X_topo
        
        for i in range(self.n_samples):
            H = self.Hs[i] #H = indicates whether the hidden var is active (1) or not (0)
            C = self.HBs[i] #C = indicates whether the current image contains a horizontal bar (1) or not (0)

            if C==0 and H==0:
                gt_labels[i] = 0 # e.g. p(T|C,H) = 0.1
            if C==0 and H==1:
                gt_labels[i] = 1 # p(T|C,H) = 0.4
            if C==1 and H==0:
                gt_labels[i] = 2 # P(T|C,H) = 0.7
            if C==1 and H==1:
                gt_labels[i] = 3 # P(T|C,H) = 1.

        return gt_labels 

    def viewImages(self):       
        n_cols = 10 
        n_rows = (self.n_samples //10) + 1


        fig, a = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False, figsize=(10,10)) 
        for i in range(self.n_samples): 
            col = i % 10 
            row = i // 10 
            a[row][col].imshow(self.X_topo[i])

            current_class = self.gt_labels[i]
            a[row][col].title.set_text(str(int(current_class)))

        for row in range(n_rows): 
            for col in range(n_cols): 
                a[row][col].axis('off')
        
        fig.tight_layout()
        plt.show()


    def generate_target(self): 
        p_dict = {0: 0.1, 1: 0.4, 2: 0.7, 3: 1.}
        
        target_vals = np.zeros(self.n_samples)

        for i in range(self.n_samples): 
            currentP = self.gt_labels[i][p_dict]
            target_vals[i] = (np.random.random() < currentP)
        return target_vals
        
