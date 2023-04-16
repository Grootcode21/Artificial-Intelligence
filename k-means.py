# NB: this requires the 'pillow' package to be installed
from sklearn.datasets import load_sample_image
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
import numpy as np

china = load_sample_image("china.jpg")

# ax = plt.axes(xticks=[], yticks=[])
# ax.imshow(china)

print(china.shape)

data = china / 255.0  # use 0...1 scale
data = data.reshape(427 * 640, 3)
print(data.shape)


def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
        # choose a random subset
        rng = np.random.RandomState(0)
        i = rng.permutation(data.shape[0])[:N]
        colors = colors[i]
        R, G, B = data[i].T

        fig, ax = plt.subplots(1, 3, figsize=(16, 6))
        ax[0].scatter(R, G, color=colors, marker='.')
        ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

        ax[1].scatter(R, B, color=colors, marker='.')
        ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

        fig.suptitle(title, size=20)


plot_pixels(data, title='Input color space: 16 million possible colors')
plt.show()

import warnings;

warnings.simplefilter('ignore')

from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

plot_pixels(data, colors=new_colors,
            title="Reduced color shape: 16 colors")
plt.show()
china_recolored = new_colors.reshape(china.reshape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-color image', size=16)
plt.show()
