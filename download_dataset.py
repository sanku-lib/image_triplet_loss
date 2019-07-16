import os
import requests
from zipfile import ZipFile

dataset_url = "http://aws-proserve-data-science.s3.amazonaws.com/geological_similarity.zip"
filePath = './data_repository/geological_similarity.zip'

if os.path.exists(filePath):
    os.remove(filePath)
else:
    print("Have to download dataset.")


r = requests.get(dataset_url, stream = True)
print('Started downloading dataset...')
with open(filePath, "wb") as data:
    for chunk in r.iter_content(chunk_size=1024):
        # writing one chunk at a time to data file

        if chunk:
            print('...',end = ''),
            data.write(chunk)
print('Download finished.')
print('Unzipping File...')
zf = ZipFile(filePath, 'r')
zf.extractall('./data_repository/')
zf.close()
print('Successfully unzipped file. Ready to run model...')