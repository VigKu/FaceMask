# !pip install bs4
# !pip install lxml
from bs4 import BeautifulSoup as bs
import pandas as pd
import os, sys
from sklearn import preprocessing


path = "C:/Users/vigne/PycharmProjects/FaceMask/annotations/"
dirs = os.listdir(path)
img_files = []
files = []

### Gather all .xml file paths ###
for file in dirs:
    files.append(file)
    img_files.append(path+file)

### Parsing ###
img_dim = []
bndbox_dim = []
class_str = []
filename = []
for i,each in enumerate(img_files):
    file = open(each)
    page = file.read()
    soup = bs(page, "xml")

    faces = soup.findAll('object')
    for face in faces:
        # append filename
        filename.append(files[i][:-4])  # remove .xml from filename

        # parse image dim
        size = soup.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        img_dim.append([width, height])

        # parse class string -> class name
        cat = face.find('name').text
        class_str.append(cat)

        # parse bndbox range
        box = face.find('bndbox')
        xmin = int(box.find('xmin').text)
        xmax = int(box.find('xmax').text)
        ymin = int(box.find('ymin').text)
        ymax = int(box.find('ymax').text)

        # format -> xmin ymin xmax ymax
        bndbox_dim.append([xmin, ymin, xmax, ymax])

### Test parsing ###
# print(img_dim[0:5])
# print(bndbox_dim[0:5])

### Create Dataframe ###
# Combine all individual lists to one list
data_table = [filename,img_dim,bndbox_dim,class_str]
# Create a dataframe using pandas
df = pd.DataFrame(data_table)
# Add column headers
df = df.transpose()
headers = ['Filename','ImgDim','Bndbox','ClassStr']
df.columns = headers

### Labeling ###
# Classify the ClassStr into 3 classes in integers
le = preprocessing.LabelEncoder()
df['label'] = le.fit_transform(df['ClassStr'])
# print(df.head())

### Save the dataframe to a csv file ###
df.to_csv("C:/Users/vigne/PycharmProjects/FaceMask/data_table.csv")
