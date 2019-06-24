import os 
import os.path as path
files = os.listdir(path.abspath(path.join(__file__,"../../dataset/")))
files = list(filter(lambda a: a != 'models' and a != 'BLOSUM50', files))


print(files)
