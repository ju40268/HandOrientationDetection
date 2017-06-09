import os
print os.path.join(os.path.sep, 'a', 'b', 'c')
relative_path = os.path.join('a','b','..', 'd')
print os.path.abspath(relative_path)
print os.getcwd() + '\\hand_raw_data\\'