import os
from pathlib import Path
import shutil
# assign directory
def reshapeMarket(directory):

	# iterate over files in
	# that directory

	for id in os.listdir(directory):
		dir = os.path.join(directory,id, "uv_maps")
		for uvmap in os.listdir(dir):
			f = os.path.join(dir, uvmap)
			if os.path.isfile(f):
				dir2 = os.path.join(directory,"uv_maps")
				if not os.path.isdir(dir2):
					Path(dir2).mkdir(exist_ok=True, parents=True)  
				new_file = os.path.join(dir2,uvmap)
				shutil.copyfile(f,new_file)

def reshapeAxis(directory,idx):
	idx = 3000
	for id in os.listdir(directory):
		dir = os.path.join(directory,id, "uv_maps")
		for uvmap in os.listdir(dir):
			f = os.path.join(dir, uvmap)
			if os.path.isfile(f):
				dir2 = os.path.join(directory,"uv_maps")
				if not os.path.isdir(dir2):
					Path(dir2).mkdir(exist_ok=True, parents=True)  
				new_file = os.path.join(dir2,f"{idx}_"+uvmap)
				shutil.copyfile(f,new_file)
		idx += 1
	print(idx)
    

# reshapeMarket('SegmentedMarket1501train/')   
#change idx to get new ids for each identityls /etc | wc -l 
reshapeAxis("/home/bjornel/Axis/ICATrain/",idx = 3000)