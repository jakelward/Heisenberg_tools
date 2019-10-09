#This script will use a timeline file which contains information to arrange fits images according to their
#position along a timeline in order to make a movie.
#
#This script makes use of img_scale.py, written by Min-Su Shin, Department of Astrophysical Sciences, Princeton University (https://astromsshin.github.io/science/code/Python_fits_image/index.html)
#img_scale.py is available at http://www.sciserver.org/wp-content/uploads/2016/04/img_scale.py_.txt
#Notes to the user:
#The timeline file MUST take the form (including a 1 line header):
#file, t_start,t_end,scale_min,scale_max, tracer name
#lmc_hi_Karl_fcalJy_rmpSHASSA.fits, 0, 59, 0, 3.28789e+19, HI
#
#the scale_min and scale_max values are used to set the brightness of each image using linear scaling along that range.
#Currently only linear scaling is used but this may be updated in the future

#Future features that are not yet implemented:
#non-linear scaling functions
#Specification of preferred image colours
#Cleanup feature to remove the output png images

#Optimisations to be implemented:
#Only load each fits file once
#Only scale images when scaling changes.

import os
import numpy as np
import pyfits
import pylab as py
from matplotlib.patches import Rectangle
import img_scale
import matplotlib.pyplot as plt
from scipy import ndimage
import matplotlib.gridspec as gridspec
#Input parameters:
image_dir = '/Users/jake/movieFiles/img_dir/'#The directory containing the images that need to be stitched together
timeline_file = '/Users/jake/movieFiles/movie20190819.dat'#A file containing all of the information for the timeline
outimg_dir = '/Users/jake/movieFiles/img_out/' #Warning! If cleanup is set to "True", then it is best to use a new, empty directory to avoid losing important data.
output_movie = 'testMovie.mp4' #Path of the output mp4 file
framerate = 24 #Framerate of movie in FPS. If tstretch == framerate, then movie represents 1Myr/s.
tstretch = 24#increase the length of the movie by this factor (must be of type int!). If tstretch == 1, then 1 frame = 1Myr
rotationPerMyr = 0.82#The speed of rotation of the LMC per Myr. Set to 0 if no rotation wanted
FadeTracersEnd = True #If set to true, this will fade each of the tracers out over 1 second
FadeTracersStart = True #If set to true, this will fade each of the tracers in over 1 second
openMovieOnClose = True #Opens the movie when it is ready.
EndAtCorrectRot = True #If set to true then the galaxy will start rotated and will end at it's observed state at the end of the video
Cleanup = True #At the end of execution, remove all of the png files from the outimg_dir.
#END OF INPUT#

#Define all necessary functions:
def arrangeTimeline(input_file=timeline_file, tstretch=tstretch):#Let's assume that we have some file in which we store three columns: 0 - a list of file names, 1 - a start time for that file, 2 - an end time for that file
    file_names = np.genfromtxt(input_file, delimiter=',', dtype=str, skip_header=1)[:,0] #I'll assume for now that there will be some sort of descriptive column header
    times = np.genfromtxt(input_file, delimiter=',', dtype=float, skip_header=1)[:,1:3] #So this should be an Nx2 array in which we have the start time at [n,0] and the end time at [n,1]
    #And now I need a three x n_timestep array that contains the three images that will define each frame in the movie.
    #Then I need to do the difficult bit: populating that array
    total_time = times[-1,1] - times[0,0]
    timesteps = int(total_time * tstretch) #We calculate the number of timesteps that will be needed
    movie_input_array = np.empty((timesteps,3),dtype='U256')
    stepimages = [None,None,None] #numpy array of three strings that can be up to 256 characters in length
    nextSlot = 0 #This is just an index that will help arrange the colours
    for i in range(timesteps):
        current_time = float(i) / float(tstretch) #Calculate the current time, ensuring that we use floating point arithmetic
        for j in range(len(stepimages)):
            curr = stepimages[j]
            for k in range(len(file_names)):
                if curr == file_names[k]: #Here we check whether we should keep this tracer in the video
                    if current_time > times[k,1]:
                        stepimages[j] = None #set to empty string
        for j in range(len(file_names)):
            if (file_names[j] not in stepimages) and (current_time >= times[j,0]) and (current_time <= times[j,1]):
                if stepimages[0] == None:
                    stepimages[0] = file_names[j]
                elif stepimages[1] == None:
                    stepimages[1] = file_names[j]
                elif stepimages[2] == None:
                    stepimages[2] = file_names[j]
                else:
                    print("No room for file " + file_names[j] + " at time " + str(current_time))
        movie_input_array[i,0] = stepimages[0]
        movie_input_array[i,1] = stepimages[1]
        movie_input_array[i,2] = stepimages[2]
    return movie_input_array

    
def assemble3colImage(imgRed,imgGreen,imgBlue,rotAngle,outname,tracerR,tracerG,tracerB,currentTime):
    if type(imgRed) == np.ndarray:
        img = np.zeros((imgRed.shape[0], imgRed.shape[1], 3), dtype=float)
    elif type(imgGreen) == np.ndarray:
        img = np.zeros((imgGreen.shape[0], imgGreen.shape[1], 3), dtype=float)
    elif type(imgBlue) == np.ndarray:
        img = np.zeros((imgBlue.shape[0], imgBlue.shape[1], 3), dtype=float)
    else:
        print("something has gone a bit wrong")
        img = np.zeros((100,100,3), dtype=float)
    if type(imgRed) == np.ndarray:
        k_img = ndimage.rotate(imgRed,rotAngle,reshape=False,order=1)
        img[:,:,0] = k_img
    if type(imgGreen) == np.ndarray:
        h_img = ndimage.rotate(imgGreen,rotAngle,reshape=False,order=1)
        img[:,:,1] = h_img
    if type(imgBlue) == np.ndarray:
        j_img = ndimage.rotate(imgBlue,rotAngle,reshape=False,order=1)
        img[:,:,2] = j_img
    fig = plt.figure() 
    gridspec.GridSpec(10,10) 
    titlestring = str(str('R=') + str(tracerR)+str(', G=')+str(tracerG)+str(', B=')+str(tracerB))
    ax1 = plt.subplot2grid((10,10), (0,0), colspan=10, rowspan=8)
    #ax1 = fig.add_subplot(2,1,1)
    ax1.imshow(img, aspect='equal',origin='lower')
    ax1.title.set_text(titlestring)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax2 = plt.subplot2grid((10,10), (8,1), colspan=8, rowspan=2)
    #ax2 = fig.add_subplot(2,1,2)
    ax2 = drawtimeline(ax2,timeline_file,tracerR,tracerG,tracerB,currentTime)
    fig.tight_layout()
    fig.savefig(outname,dpi=300)
    plt.close(fig)

def saveAsMovie(imgdir,framerate,output_movie):
    #ffmpeg -r 5 -i rgb_%01d.png -vcodec mpeg4 -y movie.mp4
    stringout = str("ffmpeg -r "+str(framerate)+" -i "+str(imgdir)+"rgb_%01d.png -vcodec mpeg4 -q:v 2 -y "+str(output_movie))
    print(stringout)
    os.system(stringout)

def MakeMovieImages(Ntimesteps, imagesDir, outputDir,movie_array,file_names,stretchValues,tracerNames,rotationPerMyr=rotationPerMyr,tstretch=tstretch):
    print('making images...')
    for i in range(Ntimesteps):
        output_name = str(outputDir+'rgb_'+str(i)+'.png')
        currentTime = (float(i) / float(tstretch))
        if EndAtCorrectRot:
            rot_angle = (-1.*(Ntimesteps-1)*(rotationPerMyr/tstretch))+(i*rotationPerMyr/tstretch)
        else:
            rot_angle = (i*rotationPerMyr/tstretch)
        if '.fits' in (movie_array[i,0]):
            stindex = np.where(file_names == movie_array[i,0])
            tracerR = (tracerNames[stindex])[0]
            imgR = img_scale.linear(pyfits.getdata(str(imagesDir+movie_array[i,0])),scale_min=stretchValues[stindex,0],scale_max=stretchValues[stindex,1])
            if FadeTracersStart:
                try:
                    if movie_array[i-framerate,0] != movie_array[i,0]:
                        stretchValues[stindex,1] = stretchValues[stindex,1]/1.1
                except:
                    print("Fade start has failed")
            if FadeTracersEnd:
                try:
                    if movie_array[i+framerate,0] != movie_array[i,0]:
                        stretchValues[stindex,1] = stretchValues[stindex,1]*1.1
                except:
                    print("Nearing end of the movie")
        else:
            imgR = None
            tracerR = 'N/A'
        if '.fits' in (movie_array[i,1]):
            stindex = np.where(file_names == movie_array[i,1])
            imgG = img_scale.linear(pyfits.getdata(str(imagesDir+movie_array[i,1])),scale_min=stretchValues[stindex,0],scale_max=stretchValues[stindex,1])
            tracerG = (tracerNames[stindex])[0]
            if FadeTracersStart:
                try:
                    if movie_array[i-framerate,1] != movie_array[i,1]:
                        stretchValues[stindex,1] = stretchValues[stindex,1]/1.1
                except:
                    print("Fade start has failed")
            if FadeTracersEnd:
                try:
                    if movie_array[i+framerate,1] != movie_array[i,1]:
                        stretchValues[stindex,1] = stretchValues[stindex,1]*1.1
                except:
                    print("Nearing end of the movie")
        else: 
            imgG = None
            tracerG = 'N/A'
        if '.fits' in (movie_array[i,2]):
            stindex = np.where(file_names == movie_array[i,2])
            tracerB = (tracerNames[stindex])[0]
            imgB = img_scale.linear(pyfits.getdata(str(imagesDir+movie_array[i,2])),scale_min=stretchValues[stindex,0],scale_max=stretchValues[stindex,1])
            if FadeTracersStart:
                try:
                    if movie_array[i-framerate,2] != movie_array[i,2]:
                        stretchValues[stindex,1] = stretchValues[stindex,1]/1.1
                except:
                    print("Fade start has failed")
            if FadeTracersEnd:
                try:
                    if movie_array[i+framerate,2] != movie_array[i,2]:
                        stretchValues[stindex,1] = stretchValues[stindex,1]*1.1
                except:
                    print("Nearing end of the movie")
        else:
            imgB = None
            tracerB = 'N/A'
        assemble3colImage(imgR,imgG,imgB,rot_angle,output_name,tracerR,tracerG,tracerB,currentTime)

def drawtimeline(axis2,timeline_file,traR,traG,traB,currentTime):
    #This function is going to draw a box for each tracer representing it's position and length along the timeline
    #There will also be a marker that moves along the timeline, showing the current position in time
    #First we need to read in the relevant data:
    timelineData = np.genfromtxt(timeline_file,delimiter=",",dtype=float,skip_header=1)
    timelineDataString = np.genfromtxt(timeline_file,delimiter=",",dtype=str,skip_header=1)
    tracerNames = timelineDataString[:,5]
    tracertimes = timelineData[:,1:3]
    #Next we need to setup some boundaries on where the timeline is then going to go
    axis2.axis(xmin=np.min(tracertimes),xmax=np.max(tracertimes))
    axis2.axis(ymin=0,ymax=len(tracerNames))
    axis2.set_xlabel("time [Myr]")
    #axis2.axis(xlabel="time [Myr]")
    axis2.set_yticks([])
    for i in range(len(tracerNames)):
        if tracerNames[i] == traR:
            rect = Rectangle((tracertimes[i,0],len(tracertimes)-(i+1)),tracertimes[i,1]-tracertimes[i,0],1,color="red",ec="black",ls="-", label=tracerNames[i])
        elif tracerNames[i] == traG:
            rect = Rectangle((tracertimes[i,0],len(tracertimes)-(i+1)),tracertimes[i,1]-tracertimes[i,0],1,color="green",ec="black",ls="-", label=tracerNames[i])
        elif tracerNames[i] == traB:
            rect = Rectangle((tracertimes[i,0],len(tracertimes)-(i+1)),tracertimes[i,1]-tracertimes[i,0],1,color="blue",ec="black",ls="-", label=tracerNames[i])
        else:
            rect = Rectangle((tracertimes[i,0],len(tracertimes)-(i+1)),tracertimes[i,1]-tracertimes[i,0],1,color="gray",ec="black",ls="-", label=tracerNames[i])
        axis2.add_patch(rect)
        axis2.text(np.min(tracertimes)-(np.max(tracertimes)/6.), len(tracertimes)-(i+1), tracerNames[i], fontsize=8)
        axis2.text(np.max(tracertimes)+(np.max(tracertimes)/35.), len(tracertimes)-(i+1), tracerNames[i], fontsize=8)
        axis2.axvline(x=currentTime,linewidth=1, color='black', ls='--')
    return axis2

def main():
#Actually execute all of the above:
    movie_array = arrangeTimeline(timeline_file,tstretch)
    arrayShape = np.shape(movie_array)
    Nsteps = arrayShape[0]
    file_names = np.genfromtxt(timeline_file, delimiter=',', dtype=str, skip_header=1)[:,0]
    tracerNames = np.genfromtxt(timeline_file, delimiter=',', dtype=str, skip_header=1)[:,5]
    stretchValues = np.genfromtxt(timeline_file, delimiter=',', dtype=float, skip_header=1)[:,3:5]
    if FadeTracersStart:
        for i in range(framerate):
            stretchValues = stretchValues * 1.1
    MakeMovieImages(Nsteps,image_dir,outimg_dir,movie_array,file_names,stretchValues,tracerNames)
    saveAsMovie(outimg_dir,framerate,output_movie)
    if Cleanup:
        cleanupstring = str("rm "+str(outimg_dir)+"rgb*.png")
        os.system(cleanupstring)
    if openMovieOnClose:
        os.system(str('open -a QuickTime\ Player ' + output_movie))

if __name__ == "__main__":
    main()
