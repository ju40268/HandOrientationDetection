import detect
import os
import file_operation

#import test_module
# main function start here!
'''
TODO:
(1)import the data file path
(2)robust way: down-sampling to filter out the abnormal angles
(3)[checked!]run file all-in-one --> traverse all the file
(4)[Checked!]log out both the timestamp and the angle to a csv file --> calculate the accuracy
(5)fix the file path problems(in the relative path, if the directory not exist, create one)
(6)[checked!]do not set up the end point manually
'''
'''
:param: left to be tuned.
'''

if __name__ == "__main__":
    f = file_operation.pwd() + 'Output_white_noise.txt'
    # file_operation.img_output()
    #print 'Now processing with: ' + f
    detect.main_online(f, header_padding=0, touchpad_center=[14, 13])

