import numpy as np
import math
from PIL import Image
import cv2
import os
import glob
import random

class DataAugmentation:
    
    def __init__(self, path = 'C:/Users/1611204/Desktop/Workshop/IMG250/'):
        self.path=path
        
    def addImg(self, batch_size = 1):
        iteration = 1
        filesUsed = 0
        files = glob.glob(self.path + '*.*')
        filesLen = len(files)
        files = random.sample(files, filesLen)
        if(batch_size <= (filesLen*7)):
            for filenames in files:
                rest = iteration/batch_size
                if((batch_size-iteration)*(filesLen-filesUsed) > 7):
                    nbtimeflip = math.ceil((batch_size-iteration+1)/(filesLen-filesUsed))
                else : 
                    nbtimeflip = batch_size-iteration+1
                
                for i in range(1,nbtimeflip+1):
                    img = Image.open(filenames)
                    img = self.flipImg(img,i)
                    img.save(os.path.splitext(filenames)[0]+'flip'+str(i)+'.jpg')
                    #flip image
                iteration = iteration + nbtimeflip
                filesUsed = filesUsed + 1
                if(iteration == batch_size+1):
                    break
            
        else: 
            print ("batch_size is to high, try with a smaller number")
    
    def flipImg(self,image,iteration):
        if(iteration == 1):
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        if(iteration == 2):
            return image.transpose(Image.ROTATE_90)
        if(iteration == 3):
            return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
        if(iteration == 4):
            return image.transpose(Image.ROTATE_180)
        if(iteration == 5):
            return image.transpose(Image.ROTATE_180).transpose(Image.FLIP_LEFT_RIGHT)
        if(iteration == 6):
            return image.transpose(Image.ROTATE_270)
        if(iteration == 7):
            return image.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)
        return image

