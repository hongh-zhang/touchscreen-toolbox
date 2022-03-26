import sys
import os
import shutil

REMOVE = ['results', 'DLC']

for (folder_path, _, files) in list(os.walk('tests')):
    if os.path.basename(folder_path) in REMOVE:
        shutil.rmtree(folder_path)
print('done')