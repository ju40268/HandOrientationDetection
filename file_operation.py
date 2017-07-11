import shutil
import os
from os.path import expanduser

source = 'D:/CDSoft/Temp/devio/build/Bazinga/'
destination = os.getcwd() + '\\raw_HID_output\\'
#current worling directory + hand_raw_data

def move(filename='Output.txt'):
    print('moving files...')
    shutil.move(source+filename, destination+filename)

def traverse():
    print('traversing all the file...')
    file_list = []
    for filename in os.listdir(destination):
        if filename.endswith(".txt"):
            print(filename)
            file_list.append(filename)
    return file_list

def remove(filename='Output.txt'):
    print('removing files in Bazinga...')
    os.remove(destination+filename)

def pwd():
    return destination

def data_output(type):
    if type == 'img':
        return os.path.join(expanduser("~"), 'Desktop', 'Contour_output','img','')
    elif type == 'csv':
        return os.path.join(expanduser("~"), 'Desktop', 'Contour_output', 'csv','')
    elif type == 'filtered':
        return os.path.join(expanduser("~"), 'Desktop', 'Contour_output', 'filtered', '')
    else:
        return os.path.join(expanduser("~"), 'Desktop', 'Contour_output', 'timestamp_angle_pair','')


