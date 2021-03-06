##!/usr/bin/env python
## coding=utf-8
import jieba

filePath='data/word2.txt'
fileSegWordDonePath ='text2.txt'
# read the file by line
fileTrainRead = []
#fileTestRead = []
with open(filePath) as fileTrainRaw:
    for line in fileTrainRaw:
        fileTrainRead.append(line)


# define this function to print a list with Chinese
def PrintListChinese(list):
    for i in range(len(list)):
        print(list[i])

# segment word with jieba
fileTrainSeg=[]
for i in range(len(fileTrainRead)):
    #fileTrainSeg.append([' '.join(list(jieba.cut(fileTrainRead[i][9:-11],cut_all=False)))])
    fileTrainSeg.append([' '.join(list(jieba.cut(fileTrainRead[i], cut_all=False)))])
    if i % 100 == 0 :
        print(i)

# to test the segment result
#PrintListChinese(fileTrainSeg[10])

# save the result
with open(fileSegWordDonePath,'wb') as fW:
    for i in range(len(fileTrainSeg)):
        fW.write(fileTrainSeg[i][0].encode('utf-8'))