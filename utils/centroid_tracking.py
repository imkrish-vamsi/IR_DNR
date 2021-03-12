# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker():
    def __init__(self, maxDisappeared=50, minArea=5000):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.minArea = minArea
        self.areaOfObjects = OrderedDict()
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        object_id = self.nextObjectID
        self.objects[object_id] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        return object_id

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.areaOfObjects[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            ids = list(self.disappeared.keys())
            for objectID in ids:
                self.disappeared[objectID] += 1
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            # return early as there are no centroids or tracking info
            # to update
            #! put the value in a list because we will use all()
            #! and I used True to mimic that all the rects have been seen before
            return [True]  # if len(ids) > 0 else [False]

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        #! bool list to check if this rect was seen before or not
        #! begin with all rects seen before
        was_seen_before = [True] * len(rects)
        areas = [0]*len(rects)  # ! area of every rect

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

            #! record te area of this rect
            areas[i] = (endX - startX) * (endY - startY)
        print(areas)
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                object_id = self.register(inputCentroids[i])
                new_area = areas[i]
                self.areaOfObjects[object_id] = new_area

                if new_area > self.minArea:
                    was_seen_before[i] = False

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it because the rows or cols might have repeated values
                # for example rows = [0,1,0] , cols = [2,1,1]
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                #! area of this object is less than area of the input rect
                old_area = self.areaOfObjects[objectID]
                new_area = areas[col]
                if old_area < new_area:
                    #! input rect area is bigger than the minArea
                    #! and also old area less than the minArea and this last logic
                    #! is every important because if both old and new area are
                    #! bigger than the minArea then we don't mark this rect as a new detection
                    #! and we detect the new rect as the same old object
                    if new_area > self.minArea and old_area < self.minArea:
                        print(f"\n[TRACKER]: NEW_AREA: {new_area}")
                        #! mark this a new object because it has a bigger and better area
                        #! so it will have better resolution
                        was_seen_before[col] = False

                    #! assign this new bigger area to the object
                    self.areaOfObjects[objectID] = new_area

                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    object_id = self.register(inputCentroids[col])
                    new_area = areas[col]
                    self.areaOfObjects[object_id] = new_area

                    if new_area > self.minArea:
                        was_seen_before[col] = False

        # return the bool list of seen plates
        return was_seen_before

    def track(self, det):
        if det is None or len(det) == 0:
            return self.update([])

        rects = [map(int, xyxy) for *xyxy, _, _ in det]
        return self.update(rects)
