# pylint: disable=C0103,W0105,W0611,C0116,C0114,C0115,C0111, C0123

from math import floor
import numpy as np
import matplotlib.pyplot as plt

# Gaussian kernel
def gaussianKernel(sigma:int, size:int=None):
    """Create a Gaussian kernel of size (size x size) and standard deviation sigma.
    """
    
    if size is None:
        # Calculate the size of the kernel
        size = 6*sigma + 1
    
    # Create a 2D array of zeros
    kernel = np.zeros((size, size))
      
    K = 1/(2*np.pi*(sigma**2))
    # Apply the Gaussian filter
    for i in range(size):
        for j in range(size):
            s = i-(size//2)
            t = j-(size//2)
            kernel[i, j] = K*np.exp(-((s**2)+(t**2))/(2*(sigma**2)))
    
    # Area under the curve should be 1, but the discrete case is only
    # an approximation, correct it
    kernel = kernel / np.sum(kernel)
    
    # Return the kernel
    return kernel


Kernel = gaussianKernel(1,3)
print(Kernel)
plt.imshow(Kernel, cmap='gray')
plt.show()
