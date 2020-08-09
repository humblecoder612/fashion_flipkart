import pandas as pd 

site_path=r'C:\Users\yash chaudhary\Desktop\flipkart\2020-08-09'
trendy_path=r'C:\Users\yash chaudhary\Desktop\flipkart\static\images'


import os
import glob

import os
from glob import glob
site_sub_folder = [ f.path for f in os.scandir(site_path) if f.is_dir() ]
trendy_sub_folder = [ f.path for f in os.scandir(trendy_path) if f.is_dir() ]
import os
import glob2
li=[]
all_sites=pd.DataFrame(li)
all_trends=all_sites
for path in site_sub_folder:
    temp=pd.read_csv(path+'\\images\\.csv')
    all_sites=pd.concat([all_sites,temp],axis=0)

for path in trendy_sub_folder:
    try:
        temp=pd.read_csv(path+'\\.csv')
        all_trends=pd.concat([all_trends,temp],axis=0)
    except:
        continue

all_trends.to_csv('trend_final.csv',index=False)
all_sites.to_csv('sites_final.csv',index=False)


