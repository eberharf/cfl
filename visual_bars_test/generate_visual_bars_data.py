#Jenna Kahn
#adapted from dataset_binary_gratings.py (Chalupka 2015)

import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

########
# A binary image dataset created from the following probabilities:
# (H is a "hidden variable", VB is "vertical bars" 
#     and HB stands for "horizontal bars".)

# Causal graph: 
# H is a binary variable that, when on, causes vertical bars and increases 
# the probability of T (the target variable). Horizontal bars also increase 
# the probability of T directly, but vertical bars do not increase the probability 
# of T. Thus, H is a hidden source of confounding.  


# P(H=0) = 0.5, P(H=1) = 0.5


# Below is the 'ground truth' that CFL should attempt to recover: 
# C = the presence of horizontal bars in image 
# H = presence of hidden causal variable

# class labels and P(T) for each class:
# 0. p(T|C=0,H=0) = 0.1
# 1. p(T|C=0,H=1) = 0.4
# 2. P(T|C=1,H=0) = 0.7
# 3. P(T|C=1,H=1) = 1.


# Here are some example function calls using this class: 
# vb_data = VisualBarsData(n_samples=20, noise_lvl=0.1)
# vb_data.getImages()
# vb_data.getGroundTruth()
# vb_data.getTarget()
# vb_data.viewImages()
########

class VisualBarsData(): 

    def __init__(self, n_samples=1000, im_shape=(10,10), noise_lvl=0, set_random_seed=None):
        '''the constructor generates n_samples binary vertical bars images, 
        generates the ground labels for each image, and generates the target behavior associated 
        with each image in separate, aligned np arrays
        
        Parameters
        n_samples (int): number of images to generate 
        im_shape (tuple with two numbers): size of images to generate in pixels 
        noise_lvl (float between 0 and 1): the amount of random noise that the images should contain (default is 0)
        set_random_seed (int): Optional, if enabled sets the random generator to a specific seed, allowing reproducible random results
        '''
        assert 0 <= noise_lvl <= 1, "noise_lvl must be between 0 and 1"
        assert len(im_shape)==2, "im_shape should contain the dimensions of a 2D image"


        self.n_samples = n_samples #number of images to generate 
        self.im_shape = im_shape
        
        self.random = np.random.default_rng(set_random_seed) # create a random number generator (optionally seeded to allow reproducible results)

        #X_images = images, Hs and HBs = arrays of len n containing ground truth about var. values in each image
        self.X_images, self.Hs, self.HBs = self._generate_images(n_samples, im_shape, noise_lvl) 
        self.gt_labels = self._ground_truth_classes() #gt_labels = array of len n with 'correct' class labels for each image 
        self.target_vals = self._generate_target() #target_vals = array of len n with value of T for each image (generated probabilistically)

    def __repr__(self): 
        '''prints the binary images as np arrays when the VisualBarsData class is printed'''
        return str(self.X_images)
    
    def getImages(self): 
        return self.X_images

    def getGroundTruth(self): 
        return self.gt_labels

    def getTarget(self):
        return self.target_vals

    def getSomeImages(self, n_images, which_class):
        '''returns n visual bars images from the desired class (0=no bars, 1=vertical bars, 2 = horizontal bars, 3=both types of bars)'''
        whichImages = np.where(self.gt_labels == which_class)[0]
        if whichImages[0].shape== (0,): 
            raise ValueError("The input class label doesn't match any classes. Try 0, 1, 2, or 3 (floats)")

        try: 
            images = np.array([self.X_images[whichImages[0]]])
            for i in range(1, n_images): 
                currentIndex = whichImages[i]
                currentIm = self.X_images[currentIndex]
                images = np.vstack((images, np.array([currentIm])))
        except IndexError: 
            print("not enough images in class to return n_images")
        finally: 
            return np.around(images)


    def _generate_images(self, n_samples, im_shape, noise_lvl):
        '''
        generates binary images containing some combination of vertical 
        bars and/or horizontal bars (or neither)

        inputs: 
        n_samples = number of images to generate 
        im_shape = dimensions of each image 
        noise_lvl = amount of noise in images (can be between 0 and 1)
        '''

        # X_images = array containing each image (each val in array represents a pixel)
        X_images = self.random.random((n_samples, im_shape[0],  
                                im_shape[1])) < noise_lvl
        X_images = X_images.astype('float32')
        
        # possible values for the number of HBs and VBs
        # starting with 0 will allow for up to 1 HB and 1 VB in each image
        HB_val_poss = [0]
        VB_val_poss = [0] 

        # Select hidden variable values.
        Hs = self.random.random(n_samples)<0.5 #Hs = array containing presence/absence of hidden var for each image 

        # Select numbers of VBs and HBs.
        VBs = self.random.choice(VB_val_poss, n_samples)+Hs
        HBs = self.random.choice(HB_val_poss, n_samples)+\
                    (self.random.random(n_samples)<0.5)

        # Make images with randomly placed Gs.
        for sample_id in tqdm(range(n_samples)):
            # Place the vertical bars.
            VB_locs = self.random.choice(range(im_shape[1]), 
                                        VBs[sample_id], replace=False)
            HB_locs = self.random.choice(range(im_shape[0]), 
                                        HBs[sample_id], replace=False)
            X_images[sample_id, HB_locs, :] = 1.
            X_images[sample_id, :, VB_locs] = 1.

        return X_images, Hs, HBs


    def _ground_truth_classes(self):
        """ 
        Generates the 'ground truth' 
        classification labels for each image based on whether hidden variable is 
        active and/or horizontal bars present

        Input 
        X_images - array of binary images with some combo of horiz/vert bars 
        Hs - array, aligned with X_images, saying whether hidden var is active for each image
        HBs - array, aligned with X_images, saying whether each image contains a horiz bar or not 

        modified from behave() in ai_gratings.py (Chalupka 2015)

        """

        # THR = 0.9    #TBH I don't know what these are about 
        # C = int(np.sum(I.reshape(self.visual_reshape).sum(axis=1) >= np.floor(THR*self.visual_reshape[0]))>0)

        gt_labels = np.zeros(self.n_samples) #gt_labels = array containing "ground truth" class labels for each image in X_images
        
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

        return gt_labels.astype(int)


    def _generate_target(self): 
        '''probabilistically generates the target behavior for each image, based on the
         ground truth probabilities expressed at the top of this file'''
        p_dict = {0: 0.1, 1: 0.4, 2: 0.7, 3: 1.}
        
        target_vals = np.zeros(self.n_samples)

        for i in range(self.n_samples): 
            currentP = p_dict[self.gt_labels[i]]
            target_vals[i] = (self.random.random() < currentP)
        return target_vals
        
    def viewImages(self, x_lbls=None):
        '''shows the images from X_images in matplotlib. Can input labels to 
        associate with each image (optional), otherwise the ground truth labels 
        will be used
        
        Parameters: 
        x_lbls (1D np array): Optional Labels to show at the top of each image. If not specified, ground truth will be used 
        '''      

        #set the number of rows/columns of images to show 
        maxImages = 50  
        n_cols = 10 
        if self.X_images.shape[0] > maxImages:
            n_rows = 5 
        else: 
            n_rows = (self.n_samples //10) + 1

        # create subplot for each image 
        fig, a = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False, figsize=(10,10)) 

        #for each image to show: 
        for i in range(min(maxImages, self.n_samples)): 
            
            #show the image in the correct row/column 
            col = i % 10 
            row = i // 10 
            a[row][col].imshow(self.X_images[i])

            #use the label for the image as the title of the subplot 
            if x_lbls is not None: 
                current_class = x_lbls[i]
            else: 
                current_class = self.gt_labels[i]
            a[row][col].title.set_text(str(int(current_class)))

        #turn off axes for all the subplots 
        for row in range(n_rows): 
            for col in range(n_cols): 
                a[row][col].axis('off')
        
        fig.tight_layout()
        plt.show()

    def saveSingleImage(self, fname):
        '''chooses a random image from X_images and saves it with the name fname'''
        image = self.X_images[self.random.choice(len(self.X_images))]
        fig = plt.figure()
        plt.imshow(image)
        plt.savefig(fname)

    def saveData(self): 
        '''saves the images, ground truth, and target effects''' 
        pass