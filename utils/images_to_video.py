# -*- coding: utf-8 -*-
"""
Build video from image sequence.
"""
import cv2
from glob import glob
  
def images_to_video(folderpath,output_filename=r'output_video.avi',out_FPS=10,width=640,height=480):
    '''Converts all images in a folder into video.'''

    images=glob(folderpath+'*.jpg')
    
    # Define the codec and create VideoWriter object.
    out = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc('M','J','P','G'), 
                           out_FPS, (int(width),int(height)))
    
    for image in images:
        img=cv2.imread(image)
        out.write(img)

    cv2.destroyAllWindows()
    
if __name__=='__main__':
    folderpath=r"D:\rpi_cal_files\\"
    images_to_video(folderpath,output_filename=r'camera_cal.avi',out_FPS=10,width=640,height=480)