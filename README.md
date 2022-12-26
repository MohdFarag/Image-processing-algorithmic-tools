# Image Processing Algorithmic Tools

**An application that illustrates the image with extension BMP, DICOM, JPEG, PNG, etc. and compute image processing tools and algorithms on it.**

## Description

- User can open any image with [.bmp, .dicom, .jpeg, .png] extension.
- User can make different transformations like:
  - zooming the image with [Bilinear, Nearest-neighbor] interpolations.
  - Rotating the images with [Bilinear, Nearest-neighbor] interpolations.
  - Shearing the images vertically or horizontally.
- User can generate:
  - T shape, Square, Rectangle, Triangle etc..
- User can draw histogram for image and equalize it.
- User can apply:
  - High Pass Filters (HPF)
  - Low Pass Filters (LPF)
    - Percentiles Filters (Median filter, Max Filter, Min filter, etc.)
  - Band Pass Filters (BBF)
  - Band Reject Filters (BRF)
  - Notch Reject Filters (BRF)

## Dependencies

- **python 3.10**

**Used packages:** `pillow pydicom matplotlib numpy opencv-python`

***to install***: `pip install [python-library]` Or `pip install -r requirements.txt`

## Preview

---

## References

<https://shrishailsgajbhar.github.io/post/Image-Processing-Image-Rotation-Without-Library-Functions>
<https://towardsdatascience.com/introduction-to-image-processing-with-python-dilation-and-erosion-for-beginners-d3a0f29ad72b>
<https://python.plainenglish.io/image-erosion-explained-in-depth-using-numpy-320c01b674a8>
<https://towardsdatascience.com/image-processing-image-scaling-algorithms-ae29aaa6b36c>
<https://meghal-darji.medium.com/implementing-bilinear-interpolation-for-image-resizing-357cbb2c2722>
<https://www.geeksforgeeks.org/python-opencv-bicubic-interpolation-for-resizing-image/>
<https://medium.com/@ami25480/morphological-image-processing-operations-dilation-erosion-opening-and-closing-with-and-without-c95475468fca>
<https://www.imageprocessingplace.com/root_files_V3/image_databases.htm>
