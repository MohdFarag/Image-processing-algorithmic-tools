import numpy as np

# Add padding to image
# TODO: Add Replicate padding & Mirror padding
def addPadding(image, paddingSize, mode="same", value=0):
    if type(paddingSize) == tuple:
        xPaddingSize, yPaddingSize = paddingSize
    else:
        xPaddingSize = paddingSize
        yPaddingSize = paddingSize

    if xPaddingSize < 0:
        xPaddingSize = 0
    if yPaddingSize < 0:
        yPaddingSize = 0
    
    xAddedPadding = 2 * xPaddingSize
    yAddedPadding = 2 * yPaddingSize

    resultImage = np.zeros((image.shape[0] + xAddedPadding, image.shape[1] + yAddedPadding))


    if mode == "same":
        resultImage.fill(value)

    elif mode == "replicate":
        scaleFactor = (resultImage.shape[0])/(image.shape[0])

        for i in range(resultImage.shape[0]):
            for j in range(-yPaddingSize,yPaddingSize):

                if i < 0:
                    x = resultImage.shape[0]+i
                else:
                    x = i

                if j < 0:
                    y = resultImage.shape[1]+j
                else:
                    y = j

                print(x/scaleFactor, y/scaleFactor)
                print(image.shape[0] - 1)
                if x/scaleFactor > image.shape[0] - 1 or y/scaleFactor > image.shape[1] - 1 :
                    resultImage[i][j] = image[int(x/scaleFactor)][int(y/scaleFactor)]
                else:
                    resultImage[i][j] = image[round(x/scaleFactor)][round(y/scaleFactor)]

        for i in range(-xPaddingSize,xPaddingSize):
            for j in range(resultImage.shape[1]):
                if i < 0:
                    x = resultImage.shape[0]+i
                else:
                    x = i
                if j < 0:
                    y = resultImage.shape[1]+j
                else:
                    y = j

                if x/scaleFactor > image.shape[0] - 1 or y/scaleFactor > image.shape[1] - 1 :
                    resultImage[i][j] = image[int(x/scaleFactor)][int(y/scaleFactor)]
                else:
                    resultImage[i][j] = image[round(x/scaleFactor)][round(y/scaleFactor)]

    for i in range(xPaddingSize, resultImage.shape[0] - xPaddingSize):
        for j in range(yPaddingSize, resultImage.shape[1] - yPaddingSize):
            resultImage[i][j] = image[i-xPaddingSize][j-yPaddingSize]
            
    return resultImage

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(matrix)
print(addPadding(matrix, 2, "replicate"))