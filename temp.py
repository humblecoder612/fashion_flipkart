import pickle
import shutil

with open("data_show.txt", "rb") as fp:   # Unpickling
    data = pickle.load(fp)
target=r'C:\Users\yash chaudhary\Desktop\flipkart\static\images\\'
for first in data:
        for second in first:
            for third in second:
                print(third[1])