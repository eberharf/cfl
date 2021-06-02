

from visual_bars.generate_visual_bars_data import VisualBarsData as VB
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# generate data
vb_data = VB(n_samples=100, noise_lvl=0.03)
images = vb_data.getImages()
images_flat = np.reshape(images, (images.shape[0],np.product(images.shape[1:])))
anchor = np.expand_dims(images_flat[0],0)

# compute distances
distances = np.squeeze(euclidean_distances(images_flat, anchor))
idx = np.argsort(distances)

# plot in distance order
grid_size = int(np.sqrt(len(idx)))
fig,axs = plt.subplots(grid_size,grid_size,figsize=(50,50))
for i,ax in zip(idx,axs.ravel()):
    ax.imshow(images[i])
    ax.set_title(distances[i])
    ax.axis('off')
plt.savefig('/Users/imanwahle/Downloads/vbars', bbox_inches='tight')


