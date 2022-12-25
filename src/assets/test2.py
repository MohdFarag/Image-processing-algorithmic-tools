import cv2
import math
import numpy as np

class rotator:

    angle = 20.0
    x = 330
    y = 330

    radians = float(angle*(math.pi/180))
    img = cv2.imread('./testInputs/Lenna/lenna256x256.jpg',0)
    width,height = img.shape


    def showImg(name, self):
        cv2.imshow(name, self.img)
        self.img = np.pad(self.img, (self.height) ,'constant', constant_values=0)
        self.width,self.height = self.img.shape

    def printWH(self):
        print(self.width)
        print(self.height)

    def rotate(self):
        emptyF = np.zeros((self.width,self.height),dtype="uint8")
        emptyB = np.zeros((self.width,self.height),dtype="uint8")
        emptyBB = np.zeros((self.width,self.height),dtype="uint8")


        for i in range(self.width):
            for j in range(self.height):
                temp = self.img[i,j]
                #forward mapping
                xf = (i-self.x)*math.cos(self.radians)-(j-self.y)*math.sin(self.radians)+self.x
                yf = (i-self.x)*math.sin(self.radians)+(j-self.y)*math.cos(self.radians)+self.x
                #backward mapping should change the forward mapping to the original image
                xbb = (i-self.x)*math.cos(self.radians)+(j-self.y)*math.sin(self.radians)+self.x
                ybb = -(i-self.x)*math.sin(self.radians)+(j-self.y)*math.cos(self.radians)+self.x
                xbb = int(xbb)
                ybb = int(ybb)
                if xf < 660 and yf < 660 and xf>0 and yf > 0:
                    emptyF[int(xf),int(yf)] = temp
                else:
                    pass
                if xbb < 660 and ybb < 660 and xbb>0 and ybb > 0:
                    emptyBB[(xbb),(ybb)] = temp
                else:
                    pass
        cv2.imshow('Forward', emptyF)
        cv2.imshow('Backward', emptyBB)

def main():
    rotator.showImg('normal', rotator)
    rotator.printWH(rotator)
    rotator.rotate(rotator)
    cv2.waitKey(0)
    cv2.destroyAllWindows


if __name__ == '__main__':
    main()