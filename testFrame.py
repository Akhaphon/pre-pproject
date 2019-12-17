import cv2
import csv
import os

# Opens the Video file
cap= cv2.VideoCapture('videoTest.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# i=0
i = 0
dir = r'C:\Users\AHoys\project\pre-project\frame'
os.chdir(dir)

file = open('writeData.csv', mode='w')

def writeCsv(i):
    writer = csv.writer(file, delimiter= ',' , quotechar = '"', quoting = csv.QUOTE_MINIMAL)
    writer.writerow([f'frame{str(i)}'])
    os.chdir(dir)
    i+=1

def writCsv(i,ear):
    writer = csv.writer(file, delimiter= ',' , quotechar = '"', quoting = csv.QUOTE_MINIMAL)
    writer.writerow([f'frame{str(i)} and {str(ear)}'])
    os.chdir(dir)
    i+=1

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    # cv2.imwrite('frame'+str(i)+'.jpg',frame)
    writeCsv(i)
    i+=1

cap.release()
cv2.destroyAllWindows()
