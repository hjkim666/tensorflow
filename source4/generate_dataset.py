import numpy as np 
import os
import scipy.misc 

#SPECIFY THE FOLDER PATHS + RESHAPE SIZE + GRAYSCALE
def rgb2gray(rgb):
    if len(rgb.shape) is 3:
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    else:
        return rgb
    
    
cwd = os.getcwd()
print("Current folder is %s" % (cwd))

paths = {"flower_photos\\daisy","flower_photos\\roses","flower_photos\\sunflowers","flower_photos\\tulips"}

imgsize = [28,28]
use_gray =1 
data_name = "myData"

print("Your image should be at")
for i, path in enumerate(paths):
    print("[%d/%d] %s/%s" % (i, len(paths), cwd, path))
print("Data will be saved to %s" %(cwd+"\\data\\"+data_name+".npz")) 

#LOAD IMAGES
nclass     = len(paths)
valid_exts = [".jpg",".gif",".png",".tga", ".jpeg"]
imgcnt     = 0
for i, relpath in zip(range(nclass), paths):
#    path = cwd + "/" + relpath
    path = cwd + "\\" + relpath
    print(path)
    flist = os.listdir(path)
    for f in flist:
        if os.path.splitext(f)[1].lower() not in valid_exts:
            continue
        fullpath = os.path.join(path, f)
        currimg  = scipy.misc.imread(fullpath)
        # Convert to grayscale  
        if use_gray:
            grayimg  = rgb2gray(currimg)
        else:
            grayimg  = currimg
        # Reshape
        graysmall = scipy.misc.imresize(grayimg, [imgsize[0], imgsize[1]])/255.
        grayvec   = np.reshape(graysmall, (1, -1))
        # Save 
        curr_label = np.eye(nclass, nclass)[i:i+1, :]
        if imgcnt is 0:
            totalimg   = grayvec
            totallabel = curr_label
        else:
            totalimg   = np.concatenate((totalimg, grayvec), axis=0)
            totallabel = np.concatenate((totallabel, curr_label), axis=0)
        imgcnt    = imgcnt + 1
print ("Total %d images loaded." % (imgcnt))

#DIVIDE TOTAL DATA INTO TRAINING AND TEST SET
def print_shape(string, x):
    print ("Shape of '%s' is %s" % (string, x.shape,))
    
randidx    = np.random.randint(imgcnt, size=imgcnt)
trainidx   = randidx[0:int(3*imgcnt/5)]
testidx    = randidx[int(3*imgcnt/5):imgcnt]
trainimg   = totalimg[trainidx, :]
trainlabel = totallabel[trainidx, :]
testimg    = totalimg[testidx, :]
testlabel  = totallabel[testidx, :]
print_shape("trainimg", trainimg)
print_shape("trainlabel", trainlabel)
print_shape("testimg", testimg)
print_shape("testlabel", testlabel)    

#SAVE TO NPZ
savepath = cwd + "/data/" + data_name + ".npz"
np.savez(savepath, trainimg=trainimg, trainlabel=trainlabel
         , testimg=testimg, testlabel=testlabel, imgsize=imgsize, use_gray=use_gray)
print ("Saved to %s" % (savepath))


#[0/4] /home/user/workspace/Rasp4/capture_srv/forward
#[1/4] /home/user/workspace/Rasp4/capture_srv/right
#[2/4] /home/user/workspace/Rasp4/capture_srv/stop
#[3/4] /home/user/workspace/Rasp4/capture_srv/left


   