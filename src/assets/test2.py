import numpy as np
import matplotlib.pyplot as plt

# Get mean of image
def getMean(image:np.ndarray) -> float:
    """ Get mean of image

    Args:
        image (np.ndarray): image to get mean of

    Returns:
        float: mean of image
    """
    
    # Flatten image
    image = image.flatten()
    # Get sum of all numbers in image
    sumOfNumbers = image.sum()
    # Get mean of image
    mean = sumOfNumbers / (image.shape[0])
    
    return mean
    
# Get median of image
def getMedian(image:np.ndarray) -> float:
    """ Get median of image

    Args:
        image (np.ndarray): image to get median of

    Returns:
        float: median of image
    """
    
    # Flatten image
    image = image.flatten()
    # Sort image
    image.sort()
    # Get median of image
    median = image[len(image)//2]
    
    return median

# Get variance of image
def getVariance(image:np.ndarray) -> float:
    """ Get variance of image

    Args:
        image (np.ndarray): image to get variance of

    Returns:
        float: variance of image
    """
    
    # Flatten image
    image = image.flatten()
    # Get mean of image
    mean = getMean(image)
    # Get variance of image
    variance = ((image - mean)**2).sum() / (image.shape[0])
    
    return variance

image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("Mean: ", getMean(image))
print("Median: ", getMedian(image))
print("Variance: ", getVariance(image))