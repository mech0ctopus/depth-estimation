# -*- coding: utf-8 -*-
"""
Stacks videos vertically or horizontally.
"""
import cv2
import numpy as np

  
def stack_videos(filename1,filename2, out_FPS=10, output_filename=r'output_stack.avi',
                 stack_direction='vertical'):
    '''Stacks videos vertically or horizontally.'''

    cam1 = cv2.VideoCapture(filename1)
    cam2 = cv2.VideoCapture(filename2)
    
    if stack_direction=='vertical':
        vert_mul=2
        hor_mul=1
        axis=0
    else:
        vert_mul=1
        hor_mul=2
        axis=1
        
    # Define the codec and create VideoWriter object.
    out = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc('M','J','P','G'), 
                          out_FPS, (int(hor_mul*cam1.get(3)),int(vert_mul*cam1.get(4))))
    
    while(cam1.isOpened()):
        ret_val1, img1 = cam1.read()
        ret_val2, img2 = cam2.read()
        
        axis
        if ret_val1 == True:
            # Write the frame into the file 'output.avi'
            vis=np.concatenate((img1, img2), axis=axis) #0:vertical, 1:horizontal
            out.write(vis)
            
        else:
            break
        
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    filename1=r"C:\Users\Craig\Documents\GitHub\depth-estimation\models\output_2011_09_26_drive_0009_unseen_10FPS.avi"
    filename2=r"C:\Users\Craig\Documents\GitHub\depth-estimation\models\2011_09_26_drive_0009_unseen_depth_10FPS.avi"
    stack_videos(filename1,filename2,stack_direction='horizontal')