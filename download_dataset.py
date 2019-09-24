import os
import requests
from zipfile import ZipFile

dataset_url = "http://aws-proserve-data-science.s3.amazonaws.com/geological_similarity.zip"
filePath = './data_repository/geological_similarity.zip'
data_directory = './data_repository'

if not os.path.exists(directory):
    try:
        os.makedir(directory)
        print(data_directory," created successfully.")
    except:
        print("Unable to create directory at ",data_directory," Please create ",data_directory," manually. Then run this file again.")

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
