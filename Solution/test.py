import cv2 as cv
import numpy as np
import glob
from main import resizeImage, showImgs, edgeDetection, findContours, drawContours, \
    leave1StHierarchy, removeSmallAreaContours, grayThreshold, closing, opening, consistentBackground

objects = [cv.imread(file) for file in glob.glob("../Training data/my-training-data/Combined objects/Test1/*.jpg")]
plasticBottleImages = [cv.imread(file) for file in glob.glob("../Training data/my-training-data/Plastic bottle/*.jpg")]

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
            # Calculate the total amount of white pixels in the first (e.g. 20) rows of the BLOB
            for countWhitePxT in range(0, nrOfRowsToExtractInfo):
                sumOfWhitePxTop += oneCntData[countWhitePxT]
            # Calculate the total amount of white pixels in the last (e.g. 20) rows of the BLOB
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

def main():
    resizedObj = resizeImage(plasticBottleImages)

    # Clear background/ even illumination
    clearBackground = []
    for img in resizedObj:
        clearBackground.append(consistentBackground(img, (200, 200)))

    # Edge detection
    grad = edgeDetectionGrad(clearBackground)
    cany = edgeDetection(resizedObj, 50, 200, 3)

    # Threshold
    mask = []
    for gray in grad:
        mask.append(grayThreshold(gray, 20))

    # Filter noise
    filter = []
    for img in mask:
        filter.append(cv.medianBlur(img, 9))

    # Morphology
    squareKernel9 = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(9, 9))
    squareKernel3 = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(3, 3))
    closingL = closing(squareKernel9, filter, 2)
    # openingL = opening(squareKernel3, closingL, 1)

    # Find contours, manipulate and draw them
    contours = findContours(closingL)[0]
    hierarchy = findContours(closingL)[1]
    oneHierarchyCnt = []
    for i in range(0, len(hierarchy)):
        oneHierarchyCnt.append(leave1StHierarchy(contours[i], hierarchy[i]))
    removeSmallAreaContours(oneHierarchyCnt, hierarchy, 300.0)

    blackMask = []
    for i in range(0, len(plasticBottleImages)):
        blackMask.append(np.zeros(resizedObj[1].shape, np.uint8))

    objDraw = drawContours(contours, blackMask, oneHierarchyCnt, (255, 255, 255), thickness=cv.FILLED)

    # allImagesCountWhitePx = countWhitePxPerRow(resizedObj, oneHierarchyCnt)
    # sumsTop = sumOfWhitePxTopAndBottomOfContour(allImagesCountWhitePx, 20)[0]
    # sumsBottom = sumOfWhitePxTopAndBottomOfContour(allImagesCountWhitePx, 20)[1]
    # ratioAllImgs = ratioWhitePxTopVsBottomPerContour(sumsTop, sumsBottom)




    showImgs('Masks', closingL)

    cv.waitKey(0)

if __name__ == '__main__':
    main()