import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("./UROPVids/thingy.png")
cropped = img[60:, 620:]

plt.imshow(cropped)
plt.show()
