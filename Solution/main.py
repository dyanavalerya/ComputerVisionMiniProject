"""
Image processing classifier for three different classes within the garbage category:
plastic bottles, cardboard, and metal cans

The project uses OpenCV Version 4.0.0, Python 3.8

To run the code make sure to change the path to the directories of the images in your
own computer (training data and test data)

Contact: Diana-Valeria Vacaru, dvacar21@student.aau.dk
"""

import cv2 as cv
import numpy as np
import glob
import math
import copy

# Load training data
cardboardImages = [cv.imread(file) for file in glob.glob("../my-training-data/Cardboard/*.jpg")]
canImages = [cv.imread(file) for file in glob.glob("../my-training-data/Cans/*.jpg")]
canImagesGray = [cv.imread(file, 0) for file in glob.glob("../my-training-data/Cans/*.jpg")]
plasticBottleImages = [cv.imread(file) for file in glob.glob("../my-training-data/Plastic bottle/*.jpg")]

# Load test data
mixedObjTest1 = [cv.imread(file, 0) for file in glob.glob("../my-training-data/Combined objects/Test1/*.jpg")]
mixedObjTest1Color = [cv.imread(file) for file in glob.glob("../my-training-data/Combined objects/Test1/*.jpg")]

# <editor-fold desc="SEGMENTATION FUNCTIONS">

def resizeImage(inputImage):
    resizedDataLocal = []
    for indexImage in inputImage:
        resizedDataLocal.append(cv.resize(indexImage, (544, 408)))
    return resizedDataLocal

# Show multiple images from file at once
def showImgs(figName, imageList):
    for i, image in enumerate(imageList):
        cv.imshow(figName + ' {}'.format(i), image)

# Threshold the RGB image to get the parts of interest
def bgrThreshold(inputImage, lowerColor, upperColor):
    localMasks = []
    for localImg in inputImage:
        localMasks.append(cv.inRange(localImg, lowerColor, upperColor))
    return localMasks

def grayThreshold(grayscale, T):
    binary = np.ndarray((grayscale.shape[0], grayscale.shape[1]), dtype='uint8')
    for i in range(0, grayscale.shape[0]):
        for j in range(0, grayscale.shape[1]):
            if grayscale[i, j] <= T:
                binary[i, j] = 0
            else:
                binary[i, j] = 255
    return binary


# Morphology
def closing(kernel, inputMasks, itr):
    localClosing = []
    for localMask in inputMasks:
        localClosing.append(cv.morphologyEx(localMask, cv.MORPH_CLOSE, kernel, iterations=itr))
    return localClosing


def opening(kernel, inputMasks, itr):
    localOpening = []
    for localMask in inputMasks:
        localOpening.append(cv.morphologyEx(localMask, cv.MORPH_OPEN, kernel, iterations=itr))
    return localOpening


def dilation(kernel, inputMasks, itr):
    localDilation = []
    for localMask in inputMasks:
        localDilation.append(cv.morphologyEx(localMask, cv.MORPH_DILATE, kernel, iterations=itr))
    return localDilation


def erosion(kernel, inputMasks, itr):
    localErosion = []
    for localMask in inputMasks:
        localErosion.append(cv.morphologyEx(localMask, cv.MORPH_ERODE, kernel, iterations=itr))
    return localErosion


# Edge detection
def edgeDetection(inputImages, threshold1, threshold2, apertureSize):
    localEdges = []
    for localImg in inputImages:
        localEdges.append(cv.Canny(localImg, threshold1, threshold2, None, apertureSize))
    return localEdges

def edgeDetectionGrad(inputImage):
    scale = 0.2
    delta = 0
    ddepth = cv.CV_16S

    grayObj = []
    for obj in inputImage:
        grayObj.append(cv.cvtColor(obj, cv.COLOR_BGR2GRAY))

    grad_x = []
    grad_y = []
    abs_grad_x = []
    abs_grad_y = []
    grad = []
    for i, gray in enumerate(grayObj):
        grad_x.append(cv.Sobel(gray, ddepth, 1, 0, ksize=5, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT))
        grad_y.append(cv.Sobel(gray, ddepth, 0, 1, ksize=5, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT))

        abs_grad_x.append(cv.convertScaleAbs(grad_x[i]))
        abs_grad_y.append(cv.convertScaleAbs(grad_y[i]))

        grad.append(cv.addWeighted(abs_grad_x[i], 0.5, abs_grad_y[i], 0.5, 0))
    return grad

# Line detection/ get slopes
def lineDetection(inputEdges, imageToGetLinesFrom):
    allSlopes = []
    for j, edgeImg in enumerate(inputEdges):
        # the slope lists will be emptied when looping through the edge data of a new image
        slopeP = []
        slope = []
        lines = cv.HoughLines(edgeImg, 1, np.pi / 180, 150, None, 0, 0)
        linesP = cv.HoughLinesP(edgeImg, 1, np.pi / 180, 65, None, 50, 10)

        # Use the probabilistic approach to get lines
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(imageToGetLinesFrom[j], (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
                if (l[2] - l[0]) != 0:
                    slopeP.append((l[3] - l[1]) / (l[2] - l[0]))

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))  # pt1 = (x1, y1)
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))  # pt2 = (x2, y2)
                cv.line(imageToGetLinesFrom[j], pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
                # Calculate the slopes of each line and add them to a list
                slope.append(
                    (int(y0 - 1000 * (a)) - int(y0 + 1000 * (a))) / (int(x0 - 1000 * (-b)) - int(x0 + 1000 * (-b))))
        # Add the slope lists for each image into a list of lists containing all images
        # Concatenate them with the lines obtained from the probabilistic approach
        allSlopes.append(slope + slopeP)
    return allSlopes


# Get the number of parallel line pairs
def findParallelLines(inputImages, slopes):
    # Create a list that will contain the number of pairs of parallel lines in each image
    # i.e. the size of the list will be the same as the amount of images from the training data
    parallelPairs = []
    # Loop through each row containing line slopes lists for one image each
    for i in range(0, len(inputImages)):
        count = 0  # reset the count of parallel lines for each image
        # Loop through the inner list of slopes and compare them to each other to find pairs
        for j in range(0, len(slopes[i])):
            slope = slopes[i][j]
            for k in range(j + 1, len(slopes[i])):
                # Check whether there are any slopes that are equal with a difference of +- 0.1
                if abs(slope - slopes[i][k]) <= 0.1:
                    count += 1
        parallelPairs.append(count)
    return parallelPairs


# Find contours of each BLOB
def findContours(inputImages):
    contours = []
    hierarchys = []
    for binaryImg in inputImages:
        iThContour, hierarchy = cv.findContours(binaryImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours.append(iThContour)
        hierarchys.append(hierarchy)
    return contours, hierarchys


# This function checks if a contour has an area smaller than 4000 px and returns True or False
def check_area(ctr, area):
    return cv.contourArea(ctr) < area


# Remove contours with very small area, considered noise
def removeSmallAreaContours(inputContours, inputHierarchies, area):
    for ith in range(0, len(inputContours)):
        countRemoved = 0
        for contour in range(0, len(inputContours[ith])):
            if check_area(inputContours[ith][contour - countRemoved], area):
                inputContours[ith].pop(contour - countRemoved)
                inputHierarchies[ith] = np.delete(inputHierarchies[ith], contour - countRemoved, 1)
                countRemoved += 1


# Leave only contours from 1st hierarchy (because in this case they define the main BLOBs)
def leave1StHierarchy(localContours, localHierarchy):
    hierarchy1Cont = copy.copy(localContours)
    countRemoved2 = 0
    for iThContour in range(0, localHierarchy.shape[1]):
        if localHierarchy[0, iThContour, 3] != -1:
            hierarchy1Cont.pop(iThContour - countRemoved2)
            countRemoved2 += 1
    return hierarchy1Cont


# Draw contours
def drawContours(inputContours, imageToDrawOn, inputHierarchy1Contour, color, thickness):
    drawContoursAllImg = []
    for i in range(0, len(inputContours)):
        # -1 is for drawing all contours at once
        drawContoursAllImg.append(cv.drawContours(imageToDrawOn[i], inputHierarchy1Contour[i], -1, color, thickness))
    return drawContoursAllImg


# Make background with even illumination
def consistentBackground(inputImage, kernel):
    mean = cv.blur(inputImage, kernel)
    clean_background = cv.subtract(mean, inputImage)
    return clean_background


# </editor-fold>

# <editor-fold desc="FEATURE EXTRACTION FUNCTIONS">

# 1. Check for circularity

def circularity(ctr):
    # Circularity = (4 * pi * Area) / Perimeter^2
    return (4 * math.pi * cv.contourArea(ctr)) / pow(cv.arcLength(ctr, True), 2)


# 2. Check for elongation (this function works only if the box lines are parallel to the image plane

def parallelBoundingBoxRatio(ctr, img):
    rect = cv.minAreaRect(ctr)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(img, [box], 0, (25, 100, 50), 2)
    yMax = np.amax(box, 0)[1]
    xMax = np.amax(box, 0)[0]
    yMin = np.amin(box, 0)[1]
    xMin = np.amin(box, 0)[0]
    width = xMax - xMin
    height = yMax - yMin
    return height / width
    # print('Max', np.amax(box, 0)[0])
    # print('Min', np.amin(box, 0)[0])


def boundingBoxRatio(ctr):
    rect = cv.minAreaRect(ctr)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    # cv.drawContours(img, [box], 0, (25, 100, 50), 2)
    maxEdge = max(rect[1][1], rect[1][0])
    if maxEdge == rect[1][0]:
        return rect[1][1] / rect[1][0]
    else:
        return rect[1][0] / rect[1][1]

# 3. Check for convexhull defects
def convexityDefects(ctr, i, img):
    # hullList = []
    depth = []
    # To be able to draw the convex hull, the coordinates of the hull points are needed
    # hull2 = cv.convexHull(ctr)
    # hullList.append(hull2)
    # cv.drawContours(img, hullList, i, (100, 200, 25), 1)
    # To be able to compute the convexity defects, the indices of contour points
    # corresponding to the hull points are needed
    hull = cv.convexHull(ctr, clockwise=True, returnPoints=False)

    defects = cv.convexityDefects(ctr, hull)
    if defects is not None:
        for j in range(defects.shape[0]):
            s, e, f, d = defects[j, 0]
            far = tuple(ctr[f][0])
            depth.append(d)
            # cv.circle(img, far, 4, [0, 0, 255], -1)

    return depth


# Calculate approximate polygon and convex hull (draw them)
def convexHull(cnt, imageToDrawOn):
    # calculate epsilon base on contour's perimeter
    # contour's perimeter is returned by cv2.arcLength
    epsilon = 0.01 * cv.arcLength(cnt, True)
    # get approx polygons
    approx = cv.approxPolyDP(cnt, epsilon, True)
    # draw approx polygons
    # cv.drawContours(imageToDrawOn, [approx], -1, (0, 255, 0), 1)

    # hull is convex shape as a polygon
    hull = cv.convexHull(cnt, returnPoints=True)
    # cv.drawContours(imageToDrawOn, [hull], -1, (0, 0, 255))

    return hull, approx

# 4. Check for compactness

def compactnessRatio(ctr, localImg):
    rect = cv.minAreaRect(ctr)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(localImg, [box], 0, (25, 100, 50), 2)
    # compactness = Area of BLOB / width * height of bounding box

    return cv.contourArea(ctr) / (rect[1][1] * rect[1][0])


# 5. Average color of objects of interest (without the background), looks at whole image
# Useful when same type of objects are in the same image or only one object

def averageColor(colorImg, field2dMask):
    # Turn 2D mask into 3D channel mask
    field3dMask = np.stack((field2dMask,) * 3, axis=-1)
    maskAndImg = cv.bitwise_and(colorImg, field3dMask)

    # filter black color and fetch color values
    data = []
    for x in range(3):
        channel = maskAndImg[:, :, x]
        indices = np.where(channel != 0)[0]
        color = np.mean(channel[indices])
        data.append(int(color))

    return data


# Average color per object of interest

def averageColorPerObject(colorImg, contours):
    meanOfColours = []
    for contour in contours:
        # Finds the average of each colour channel inside the contour
        mask = np.zeros((colorImg.shape[0], colorImg.shape[1]), np.uint8)
        cv.drawContours(mask, [contour], 0, 255, -1)
        meanOfColours = cv.mean(colorImg, mask=mask)
    return meanOfColours

# 6. Different areas within BLOB pixel count ratio

# Count number of white pixels per row per contour and create a 3-level list
# with data from contours from multiple images
def countWhitePxPerRow(originalImg, allImgContours):
    allImagesCountWhilePx = []
    # Loop through all images' contour data
    for cnts in allImgContours:
        oneImgContoursWhitePxCount = []
        # Loop through all contours of one image
        for oneCnt in cnts:
            blackMaskPerCnt = np.zeros((originalImg[0].shape[0], originalImg[0].shape[1]), np.uint8)
            # Draw a mask with only one contour at a time
            cntDraw = cv.drawContours(blackMaskPerCnt, [oneCnt], 0, 255, -1)
            # Count the amount of pixels per row of each contour (without the background)
            countWhitePxPerRow = [np.count_nonzero(row) for row in cntDraw if np.count_nonzero(row) != 0]
            # Show the singled contours
            # cv.imshow('Grad', cntDraw)
            # cv.waitKey(0)
            oneImgContoursWhitePxCount.append(countWhitePxPerRow)
        allImagesCountWhilePx.append(oneImgContoursWhitePxCount)
    return allImagesCountWhilePx

def countWhitePxPerCol(originalImg, allImgContours):
    allImagesCountWhilePx = []
    # Loop through all images' contour data
    for cnts in allImgContours:
        oneImgContoursWhitePxCount = []
        # Loop through all contours of one image
        for oneCnt in cnts:
            blackMaskPerCnt = np.zeros((originalImg[0].shape[0], originalImg[0].shape[1]), np.uint8)
            # Draw a mask with only one contour at a time
            cntDraw = cv.drawContours(blackMaskPerCnt, [oneCnt], 0, 255, -1)
            # Count the amount of pixels per row of each contour (without the background)
            countWhitePxPerCol = [np.count_nonzero(col) for col in np.transpose(cntDraw) if np.count_nonzero(col) != 0]
            # Show the singled contours
            # cv.imshow('Grad', cntDraw)
            # cv.waitKey(0)
            oneImgContoursWhitePxCount.append(countWhitePxPerCol)
        allImagesCountWhilePx.append(oneImgContoursWhitePxCount)
    return allImagesCountWhilePx


# Get the amount of white pixels at the top and bottom of a BLOB
# useful for example for recognizing bottles, because the top has less pixels than the bottom
def sumOfWhitePxTopAndBottomOfContour(listOfAllImgCountWhitePx, nrOfRowsToExtractInfo):
    sumsTop = []
    sumsBottom = []
    for imageData in listOfAllImgCountWhitePx:
        sumsPerImgTop = []
        sumsPerImgBottom = []
        for oneCntData in imageData:
            sumOfWhitePxTop = 0
            sumOfWhitePxBottom = 0
            # Calculate the total amount of white pixels in the first (e.g. 20) rows/ col of the BLOB
            for countWhitePxT in range(0, nrOfRowsToExtractInfo):
                sumOfWhitePxTop += oneCntData[countWhitePxT]
            # Calculate the total amount of white pixels in the last (e.g. 20) rows/ col of the BLOB
            for countWhitePxB in range(len(oneCntData) - nrOfRowsToExtractInfo, len(oneCntData)):
                sumOfWhitePxBottom += oneCntData[countWhitePxB]
            sumsPerImgTop.append(sumOfWhitePxTop)
            sumsPerImgBottom.append(sumOfWhitePxBottom)
        sumsTop.append(sumsPerImgTop)
        sumsBottom.append(sumsPerImgBottom)
    return sumsTop, sumsBottom


# Get the ratio of the amount of white pixels at the top of a BLOB vs at the bottom of it
def ratioWhitePxTopVsBottomPerContour(sumListTop, sumListBottom):
    ratioAllImgs = []
    for sumsPerImgTop, sumsPerImgBottom in zip(sumListTop, sumListBottom):
        ratioOfTopVsBottomOfContours = []
        for sumOfWhitePxTop, sumOfWhitePxBottom in zip(sumsPerImgTop, sumsPerImgBottom):
            maximum = max(sumOfWhitePxTop, sumOfWhitePxBottom)
            if maximum == sumOfWhitePxBottom:
                ratioOfTopVsBottomOfContours.append(round(sumOfWhitePxTop / sumOfWhitePxBottom, 2))
            else:
                ratioOfTopVsBottomOfContours.append(round(sumOfWhitePxBottom / sumOfWhitePxTop, 2))
        ratioAllImgs.append(ratioOfTopVsBottomOfContours)
    return ratioAllImgs

# </editor-fold>


def main():
    # <editor-fold desc="CARDBOARD SEGMENTATION">
    # Resize the images
    resizedCardboardData = resizeImage(cardboardImages)

    # Color thresholding
    lowerBrown = np.array([20, 31, 45])
    upperBrown = np.array([129, 147, 164])
    cardboardMasks = bgrThreshold(resizedCardboardData, lowerBrown, upperBrown)

    # Morphology
    squareKernel9 = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(9, 9))
    cardboardClosing = closing(squareKernel9, cardboardMasks, 3)

    # Edge detection
    cardboardEdges = edgeDetection(resizedCardboardData, 50, 200, 3)

    # Find line slopes
    allSlopes = lineDetection(cardboardEdges, resizedCardboardData)

    # Find parallel lines (the assumption is that cardboard has many parallel
    # lines compared to other objects because of its texture)
    parallelPairs = findParallelLines(cardboardImages, allSlopes)

    # Find contours
    cardboardContours = findContours(cardboardClosing)[0]
    cardboardHierarchys = findContours(cardboardClosing)[1]

    # Remove small areas (noise)
    removeSmallAreaContours(cardboardContours, cardboardHierarchys, 4000.0)

    # Keep only 1st Hierarchy contours
    cardboardHierarchy1Contour = []
    for i in range(0, len(cardboardHierarchys)):
        cardboardHierarchy1Contour.append(leave1StHierarchy(cardboardContours[i], cardboardHierarchys[i]))

    # Draw the contours
    cardboardDrawContours = drawContours(cardboardContours, resizedCardboardData, cardboardHierarchy1Contour, (0, 255, 255), 2)

    # </editor-fold>

    # <editor-fold desc="CAN SEGMENTATION">
    resizedCanDataColor = resizeImage(canImages)
    resizedCanDataGray = resizeImage(canImagesGray)

    # Clear background/ even illumination
    clearBackground = []
    for img in resizedCanDataGray:
        clearBackground.append(consistentBackground(img, (200, 200)))

    # Thresholding
    canMask = []
    for gray in clearBackground:
        canMask.append(grayThreshold(gray, 50))

    # Noise removal & morphology
    roundKernel3 = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(3, 3))
    roundKernel31 = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(31, 31))
    canClosing = closing(roundKernel31, canMask, 1)

    canMedianFiler = []
    for img in canClosing:
        canMedianFiler.append(cv.medianBlur(img, 31))

    # Find contours
    canContours = findContours(canMedianFiler)[0]
    canHierarchys = findContours(canMedianFiler)[1]

    # Remove small areas (noise)
    removeSmallAreaContours(canContours, canHierarchys, 3000.0)

    # Keep only 1st Hierarchy contours
    canHierarchy1Contour = []
    for i in range(0, len(canHierarchys)):
        canHierarchy1Contour.append(leave1StHierarchy(canContours[i], canHierarchys[i]))

    # Draw the contours
    canDrawContours = drawContours(canContours, resizedCanDataColor, canHierarchy1Contour, (0, 255, 255), 2)

    # </editor-fold>

    # <editor-fold desc="PLASTIC BOTTLE SEGMENTATION">
    resizedPlasticBottleData = resizeImage(plasticBottleImages)
    # </editor-fold>

    # <editor-fold desc="UNKNOWN CLASS SEGMENTATION">
    # Resize data for a faster processing
    resizedUnknownData1 = resizeImage(mixedObjTest1)
    resizedUnknownData1Color = resizeImage(mixedObjTest1Color)

    # Edge detection
    unknownEdge = edgeDetectionGrad(resizedUnknownData1Color)

    # Threshold
    unknownMask = []
    for gray in unknownEdge:
        unknownMask.append(grayThreshold(gray, 50))

    # Filter noise
    unknownFilter = []
    for img in unknownMask:
        unknownFilter.append(cv.medianBlur(img, 7))

    # Morphology
    squareKernel9 = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(9, 9))
    closingL = closing(squareKernel9, unknownFilter, 2)

    # Find contours, manipulate and draw them
    unknownContours = findContours(closingL)[0]
    unknownHierarchys = findContours(closingL)[1]

    # Remove small area contours (noise)
    removeSmallAreaContours(unknownContours, unknownHierarchys, 300.0)

    # Leave only 1st hierarchy contours
    oneHierarchyUnknownCnt = []
    for i in range(0, len(unknownHierarchys)):
        oneHierarchyUnknownCnt.append(leave1StHierarchy(unknownContours[i], unknownHierarchys[i]))

    # Create a black and white mask
    blackMask = []
    for i in range(0, len(resizedUnknownData1Color)):
        blackMask.append(np.zeros(resizedUnknownData1Color[1].shape, np.uint8))

    # Draw the contours
    unknownFilledContours = drawContours(unknownContours, blackMask, oneHierarchyUnknownCnt, (255, 255, 255), thickness=cv.FILLED)
    unknownDrawContours = drawContours(unknownContours, resizedUnknownData1Color, oneHierarchyUnknownCnt, (0, 255, 255), 2)

    # </editor-fold>

    # <editor-fold desc="SAVE FEATURES TO DATABASE">

    # Create feature lists
    # Cardboard class
    cardboardAverageColorList = []
    cardboardCompactnessRatioList = []
    for i in range(0, len(resizedCardboardData)):
        cardboardAverageColorList.append(averageColor(resizedCardboardData[i], cardboardClosing[i]))
        for j in range(0, len(cardboardHierarchy1Contour[i])):
            cardboardCompactnessRatioList.append(
                compactnessRatio(cardboardHierarchy1Contour[i][j], resizedCardboardData[i]))

    # Can class
    canElongationList = []
    canConvexityDefectsAllContours = []
    for i in range(0, len(resizedCanDataColor)):
        canConvexityDefectPerContour = []
        for j in range(0, len(canHierarchy1Contour[i])):
            canElongationList.append(boundingBoxRatio(canHierarchy1Contour[i][j]))
        for k, contour in enumerate(canContours[i]):
            canConvexityDefectPerContour.append(len(convexityDefects(contour, k, resizedCanDataColor[i])))
        canConvexityDefectsAllContours.append(canConvexityDefectPerContour)

    # Plastic bottle class
    bottleConvexityDefects = []

    # Save cardboard features to file
    cardboardClassFile = copy.copy(cardboardAverageColorList)
    for i in range(0, len(cardboardClassFile)):
        cardboardClassFile[i].append(cardboardCompactnessRatioList[i])
    np.savetxt('cardboardClass.csv', cardboardClassFile, delimiter=",")

    # Save can features to file
    np.savetxt('canCass.csv', canElongationList, delimiter=",")

    # </editor-fold>

    # <editor-fold desc="CLASSIFIER">
    mean = []
    unknownHull = []
    unknownApprox = []
    convexityDefectsList = []
    boundingBoxRatioList = []
    circularityList = []
    for i in range(0, len(resizedUnknownData1Color)):
        convexityDefect = []
        bBoxRatioCnt = []
        circ = []
        averageCol = []
        compactness = []
        mean.append(averageColorPerObject(resizedUnknownData1Color[i], oneHierarchyUnknownCnt[i]))
        for j, contour in enumerate(oneHierarchyUnknownCnt[i]):
            # Used for analyzing data and debugging
            unknownHull.append(convexHull(contour, resizedUnknownData1Color[i])[0])
            unknownApprox.append(convexHull(contour, resizedUnknownData1Color[i])[1])
            compactness = compactnessRatio(contour, resizedUnknownData1Color[i])
            coordinates = (contour[math.floor(contour.shape[0] / 4), 0, 0], contour[math.floor(contour.shape[0] / 4), 0, 1])
            # cv.putText(unknownDrawContours[i], str(round(compactness, 2)), coordinates,
            #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            convexityDefect.append(max(convexityDefects(contour, j, resizedUnknownData1Color[i])))
            # cv.putText(unknownDrawContours[i], str(round(convexityDefect[j], 2)), coordinates,
            #            cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            bBoxRatioCnt.append(boundingBoxRatio(contour))
            # cv.putText(unknownDrawContours[i], str(round(bBoxRatioCnt[j], 2)), coordinates,
            #            cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            circ.append(circularity(contour))
            # cv.putText(unknownDrawContours[i], str(round(circ[j], 2)), coordinates,
            #            cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            averageCol.append(averageColorPerObject(resizedUnknownData1Color[i], oneHierarchyUnknownCnt[i]))
            averageColRound = (round(averageCol[j][0], 2), round(averageCol[j][1], 2), round(averageCol[j][2], 2))
            # cv.putText(unknownDrawContours[i], str(averageColRound), coordinates,
            #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            compactness = compactnessRatio(contour, resizedUnknownData1Color[i])
            cv.putText(unknownDrawContours[i], str(round(compactness, 2)), coordinates,
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            # Classification
            if (135.0, 150.0, 137.0) <= averageColorPerObject(resizedUnknownData1Color[i], oneHierarchyUnknownCnt[i]) <= (223.0, 239.0, 237.0)\
                    and compactnessRatio(contour, resizedUnknownData1Color[i]) >= 0.88:
                # contour[0, 0, 0] and contour[0, 0, 1] is the x and y coordinate of a pixel from the contour
                cv.putText(unknownDrawContours[i], 'Cardboard', (contour[0, 0, 0], contour[0, 0, 1]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if 0.55 < (boundingBoxRatio(contour) < 0.85 and max(convexityDefects(contour, j, resizedUnknownData1Color[i])) < 1400
                       and averageColorPerObject(resizedUnknownData1Color[i], oneHierarchyUnknownCnt[i]) <= (164.0, 197.0, 210.0)) or circularity(contour) > 0.60:
                cv.putText(unknownDrawContours[i], 'Can', (contour[0, 0, 0], contour[0, 0, 1]),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # if (max(convexityDefects(contour, j, resizedUnknownData1Color[i]))) > 1401 and circularity(contour) < 0.6:
            #     cv.putText(unknownDrawContours[i], 'Plastic bottle', (contour[0, 0, 0], contour[0, 0, 1]),
            #                cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        convexityDefectsList.append(convexityDefect)
        boundingBoxRatioList.append(bBoxRatioCnt)
        circularityList.append(circ)

    allImagesCountWhitePxRow = countWhitePxPerRow(resizedUnknownData1Color, oneHierarchyUnknownCnt)
    sumsTop = sumOfWhitePxTopAndBottomOfContour(allImagesCountWhitePxRow, 30)[0]
    sumsBottom = sumOfWhitePxTopAndBottomOfContour(allImagesCountWhitePxRow, 30)[1]
    ratioAllImgs = ratioWhitePxTopVsBottomPerContour(sumsTop, sumsBottom)

    allImagesCountWhitePxCol = countWhitePxPerCol(resizedUnknownData1Color, oneHierarchyUnknownCnt)
    sumsTopCol = sumOfWhitePxTopAndBottomOfContour(allImagesCountWhitePxCol, 10)[0]
    sumsBottomCol = sumOfWhitePxTopAndBottomOfContour(allImagesCountWhitePxCol, 10)[1]
    ratioAllImgsCol = ratioWhitePxTopVsBottomPerContour(sumsTopCol, sumsBottomCol)

    for i, innerList in enumerate(ratioAllImgs):
        for j, data in enumerate(innerList):
            if data < 0.7:
                cv.putText(unknownDrawContours[i], 'Bottle', (oneHierarchyUnknownCnt[i][j][0, 0, 0], oneHierarchyUnknownCnt[i][j][0, 0, 1]),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for i, innerList in enumerate(ratioAllImgsCol):
        for j, data in enumerate(innerList):
            if data <= 0.50:
                cv.putText(unknownDrawContours[i], 'Bottle', (oneHierarchyUnknownCnt[i][j][0, 0, 0], oneHierarchyUnknownCnt[i][j][0, 0, 1]),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (10, 100, 10), 2)

    # </editor-fold>

    # Show results
    # for i, image in enumerate(cardboardDrawContours):
    #     cv.imshow('cardboard {}'.format(i), image)

    showImgs('Unknown objects', unknownDrawContours)

    # for i, image in enumerate(unknownDrawContours):
    #     cv.imshow('cardboard {}'.format(i), image)

    # for j, im in enumerate(resizedUnknownData1):
    #     cv.imshow('cardboard {}'.format(j + len(cl1)), im)

    cv.waitKey(0)


if __name__ == '__main__':
    main()
