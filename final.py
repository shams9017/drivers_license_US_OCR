from ast import Try
import os
from operator import indexOf
from PIL import Image
from pytesseract import pytesseract
import cv2
import re
import numpy as np
from dateutil.parser import parse
from scipy.ndimage import interpolation as inter


def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite('result.png', corrected)

def dilate(path):
    image = cv2.imread(path)
    kernel = np.ones((3,4),np.uint8)
    dilation= cv2.dilate(image, kernel, iterations = 1)
    cv2.imwrite('result.png', dilation)

def erode(path):
    image = cv2.imread(path)
    kernel = np.ones((1,1),np.uint8)
    erosion =  cv2.erode(image, kernel, iterations = 1)
    cv2.imwrite('result.png', erosion)


def denoiseImg(path,newPath):
    img = cv2.imread(path)
    result = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    cv2.imwrite(newPath, result)

def turnImgToGrayscale(path,newPath):
    # read image
    img = cv2.imread(path)

    # OTSU thresh
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # you can uncomment the two lines
    # below to see the grayscale image
    # cv2.imshow("OTSU", otsu)
    # cv2.waitKey(0)

    cv2.imwrite(newPath, otsu)

def extractText(path):
    # Defining paths to tesseract.exe\
    # your path may be different. please double check 
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    img = Image.open(path)
    pytesseract.tesseract_cmd = path_to_tesseract

    # This function will
    # extract the text from the image
    text = pytesseract.image_to_string(img)
  
    # Displaying the extracted text
    print(text[:-1])

    return text

# validates if a string is date of any format
def is_date(string, fuzzy=False):

    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False

# function gets the index of the expire date line
def getDataIndexFromList(text):
    lineArr = text.split('\n')
    indexForExpDate = 0
    breakFromLoop = False
    # remove empty element
    for item in lineArr:
        if(item == ''):
            lineArr.pop(lineArr.index(item))

    for i in range(0,len(lineArr)):
        if(breakFromLoop):
            break
        expDate = re.search(r'(\d+/\d+/\d+)', lineArr[i])
        if(expDate != None):
            expDate = expDate.group(1).strip()
            expDate = expDate.split(' ')
            for j in range(0,len(expDate)):
                if(breakFromLoop):
                    break
                if(is_date(expDate[j])):
                    if(len(expDate[j])==10): # format mm/dd/YYYY is always len 10
                        indexForExpDate = i
                        breakFromLoop = True
    return indexForExpDate

# function uses a combination of regex and other string techniques
# to extract the readable output into a dictionary
# Tip: uncomment to print lineArr which will show you what has been read
def extractToDict(text):

    lineArr = text.split('\n')

    data = {}
    indexDict = {}
    indexDict['lastNameIndex'] = 0
    indexDict['firstNameIndex'] = 0
    indexDict['addressIndex'] = 0
    indexDict['cityStatePostalIndex'] = 0
    indexDict['dobIndex'] = 0

    expDateIndex = getDataIndexFromList(text)
    try:
        counter = 1
        for key in indexDict:
            indexDict[key] = expDateIndex + counter
            counter += 1

        lastNameIndex = indexDict['lastNameIndex']
        firstNameIndex = indexDict['firstNameIndex']
        addressIndex = indexDict['addressIndex']
        cityStatePostalIndex = indexDict['cityStatePostalIndex']
        dobIndex = indexDict['dobIndex']

        # remove empty element
        for item in lineArr:
            if(item.strip() == ''):
                lineArr.pop(lineArr.index(item))

        if(lineArr[-1] == ''):
            lineArr.pop(lineArr.index(lineArr[-1]))

        #print(lineArr) # uncomment this to see the raw tesseract text in list format
        print((' '))

        # individual element extraction  
        for char in lineArr[lastNameIndex]:
            if(char.lower() == 'n'):
                data["Last Name"] = lineArr[lastNameIndex][lineArr[lastNameIndex].rfind(char)+1:]
                break
            else:
                if(char.isalpha()):
                    data["Last Name"] = lineArr[lastNameIndex][lineArr[lastNameIndex].rfind(char)+2:]
                    break

        for char in lineArr[firstNameIndex]:
            if(char.lower() == 'n'):
                data["First Name"] = lineArr[firstNameIndex][lineArr[firstNameIndex].rfind(char)+1:]
                break
            else:
                if(char.isalpha()):
                    data["First Name"] = lineArr[firstNameIndex][lineArr[firstNameIndex].rfind(char)+2:]
                    break
        if(re.search(r'(\d+/\d+/\d+)', lineArr[expDateIndex]) != None ):
            expDate = re.search(r'(\d+/\d+/\d+)', lineArr[expDateIndex]).group(1)   
            data["DL Expiry"] = expDate
        if(re.search(r'(\d{1,5} [\w\s\d]{1,13} #(\d{1,5}))', lineArr[addressIndex]) != None ):
            addressFull = re.search(r'(\d{1,5} [\w\s]{1,13} #(\d{1,5}))', lineArr[addressIndex]).group(1)
    
            streetNum = re.search(r'(\d{1,5})', addressFull).group(1)
            streetName = addressFull.split(streetNum)[1]
            data["Street Number"] = streetNum
            data["Street Name"] = streetName
        elif(re.search(r'(\d{1,5} [\w\s\d]{1,13})', lineArr[addressIndex]) != None ):
            addressFull = re.search(r'(\d{1,5} [\w\s]{1,13})', lineArr[addressIndex]).group(1)  
            streetNum = re.search(r'(\d{1,5})', addressFull).group(1)
            streetName = addressFull.split(streetNum)[1]
            data["Street Number"] = streetNum
            data["Street Name"] = streetName

        cityPostalFull = lineArr[cityStatePostalIndex]
        if(len(cityPostalFull.split(',')) == 3): # checks if the city and postal are properly read
            cityName = cityPostalFull.split(',')[0]
            stateName = cityPostalFull.split(',')[1].strip().split(' ')[0]
            postal = cityPostalFull.split(',')[1].strip().split(' ')[1]
            data["City"] = cityName
            data["State"] = stateName
            data["Postal"] = postal
        elif(len(cityPostalFull.split(',')) == 2):
            cityName = cityPostalFull.split(',')[0]
            stateName = cityPostalFull.split(',')[1]
            if(len(stateName.strip().split(' ')) == 1):
                postal = lineArr[cityStatePostalIndex+1]
                stateName = stateName.strip().split(' ')[0]       
            else:
                postal = stateName.strip().split(' ')[1]
                stateName = stateName.strip().split(' ')[0]   
                
            data["City"] = cityName
            data["State"] = stateName
            data["Postal"] = postal

        if(re.search(r'(\d+/\d+/\d+)', lineArr[dobIndex]) != None ):
            data["DOB"] = re.search(r'(\d+/\d+/\d+)', lineArr[dobIndex]).group(1)
        elif(re.search(r'(\d+/\d+/\d+)', lineArr[dobIndex+1]) != None ):
            data["DOB"] = re.search(r'(\d+/\d+/\d+)', lineArr[dobIndex+1]).group(1)
        elif(re.search(r'(\d+/\d+/\d+)', lineArr[dobIndex-1]) != None ):
            data["DOB"] = re.search(r'(\d+/\d+/\d+)', lineArr[dobIndex-1]).group(1)

        print(data)
        return data

    except ValueError:
        print("an error has occured")
    except IndexError:
        print("an IndexError has occured")
    
# takes the dict and exports the data to csv
def exportToCSV(data):
    if(data == None):
        return
    cols = ["Last Name", "First Name", "DL Expiry", "Street Number", "Street Name", "City", "State", "Postal", "DOB"]
    fileExists = os.path.exists("dl.txt")

    with open("dl.txt", 'w') as f:
        if(not fileExists):            
             f.write('"Last Name", "First Name", "DL Expiry", "Street Number", "Street Name", "City", "State", "Postal", "DOB"\n')
        for key in cols:
            if(key not in data.keys()):
                data[key] = ""
        f.write('{0},{1},{2},{3},{4},{5},{6},{7},{8}\n'.format(data["Last Name"], data["First Name"], data["DL Expiry"], data["Street Number"], data["Street Name"], data["City"], data["State"], data["Postal"], data["DOB"]))
            



#execute functions here

path = 'dl11.webp'


#erode(path)

denoiseImg(path,'result.png')
turnImgToGrayscale('result.png','result.png')
erode('result.png')
text = extractText('result.png')

data = extractToDict(text)
exportToCSV(data)




# pre-processing combinations

# dl3.jpg
# denoiseImg(path,'result.png')
# denoiseImg('result.png','result.png')
# turnImgToGrayscale('result.png','result.png')
# denoiseImg('result.png','result.png')
# text = extractText('result.png')

#dl4
# denoiseImg(path,'result.png')
# image = cv2.imread('result.png')
# correct_skew(image)
# text = extractText('result.png')

# dl2.jpg
#denoiseImg(path,'result.png')

# dl1.jpg
# denoiseImg(path,'result.png')
# denoiseImg('result.png','result.png')
# denoiseImg('result.png','result.png')
# denoiseImg('result.png','result.png')
# text = extractText('result.png')

#dl7.png
# denoiseImg(path,'result.png')
# turnImgToGrayscale('result.png','result.png')
# dilate('result.png')





