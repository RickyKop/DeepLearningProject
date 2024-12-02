import zipfile as zf
files = zf.ZipFile("Code.zip", 'r')
files.extractall('Code')
files.close()