import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

def rebuild_img(u,sigma,v,num):
     m=len(u)
     n=len(v)
     a=np.zeros((m,n))

     k=0

     while k <= num:
         uk=u[:,k].reshape(m,1)
         vk=v[k].reshape(1,n)
         a+=sigma[k]*np.dot(uk,vk)
         k+=1


     return a.astype('float32')

# This is the code to compress the given image to 50% of its original size.
im = plt.imread("D:/CMU_Grayscale.png")
rate = 0.5
num = (int) (im.shape[0] * im.shape[1] * rate / (im.shape[0] + im.shape[1]))

u,sigma,v=np.linalg.svd(im)
after_compressed = rebuild_img(u,sigma,v,num)

plt.title("The size is: " + str(rate * 100) + "% of the original size.")
plt.imshow(after_compressed,cmap = plt.cm.gray_r)
plt.show()


# This is the code to compress the given image to 10% of its original size.
im = plt.imread("D:/CMU_Grayscale.png")
rate = 0.1
num = (int) (im.shape[0] * im.shape[1] * rate / (im.shape[0] + im.shape[1]))


u,sigma,v=np.linalg.svd(im)
after_compressed = rebuild_img(u,sigma,v,num)

plt.title("The size is: " + str(rate * 100) + "% of the original size.")
plt.imshow(after_compressed,cmap = plt.cm.gray_r)
plt.show()

# This is the code to compress the given image to 5% of its original size.
im = plt.imread("D:/CMU_Grayscale.png")
rate = 0.05
num = (int) (im.shape[0] * im.shape[1] * rate / (im.shape[0] + im.shape[1]))

u,sigma,v=np.linalg.svd(im)
after_compressed = rebuild_img(u,sigma,v,num)

plt.title("The size is: " + str(rate * 100) + "% of the original size.")
plt.imshow(after_compressed,cmap = plt.cm.gray_r)
plt.show()
