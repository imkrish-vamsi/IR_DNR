import imutils
from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2

usePlateDetectEngine = True
newW = 320
newH = 320
min_confidence = 0.5

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text

layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

if usePlateDetectEngine == True:
    # load the pre-trained EAST text detector
    # print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet("pretrained/frozen_east_text_detection.pb")


def plateDetect(image, upperPath, bottomPath):
    divideRatio = 0.5
    rotated = image.copy()
    height = np.size(rotated, 0)
    width = np.size(rotated, 1)
    lastStartY = int(height * divideRatio)

    if usePlateDetectEngine == True:

        # 1. detect slope angle and rotate the image.
        orig = image.copy()
        (H, W) = image.shape[:2]
        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        yAngle = 0
        nYAngle = 0

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            xAngle = 0
            nXAngle = 0
            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < min_confidence:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                xAngle += angle
                nXAngle += 1
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

            if nXAngle > 0:
                xAngle /= nXAngle
                yAngle += xAngle
                nYAngle += 1

        if nYAngle > 0:
            yAngle /= nYAngle

        degreeAngle = yAngle * 180/np.pi
        print("degree", degreeAngle)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        rotated = imutils.rotate_bound(orig, degreeAngle)

        # 2. detect text from rotated mat.
        orig = rotated.copy()
        (H, W) = orig.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height

        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        orig = cv2.resize(orig, (newW, newH))
        (H, W) = orig.shape[:2]

        blob = cv2.dnn.blobFromImage(
            orig, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < min_confidence:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        lastStartY = 0
        square = 0

        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            isquare = (endX - startX) * (endY-startY)
            if isquare > square:
                lastStartY = startY
                square = isquare
            #cv2.rectangle(rotated, (startX, startY), (endX, endY), (0, 255, 0), 2)

        height = np.size(rotated, 0)
        width = np.size(rotated, 1)

        boxCount = np.size(boxes, 0)
        if boxCount == 0:
            lastStartY = int(height * divideRatio)
    uppermat = rotated[0:lastStartY, 0:width]
    bottommat = rotated[lastStartY:height, 0:width]
    # show the output image
    if np.sum(uppermat) != 0:
        cv2.imwrite(upperPath, uppermat)
    if np.sum(bottommat) != 0:
        cv2.imwrite(bottomPath, bottommat)
    return {'upper': uppermat, 'bottom': bottommat}
