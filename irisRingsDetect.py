import numpy as np
import pywt
import cv2
import imutils as im
def w2d(img, mode='haar', level=1):
    imArray = cv2.imread(img)
    print (imArray.shape, imArray.size)
    img = imArray.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blank = cv2.imread("/home/rahul/Pictures/blank.jpeg")
    cv2.resize(blank,(178,178))
    blank = im.resize(blank,178,178,cv2.INTER_AREA)
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    edge = cv2.Canny(thresh1, 150, 150, apertureSize=3, L2gradient=True)
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)
    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;
    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)
    # ==========
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    # Change thresholds
    params.minThreshold = 100
    params.maxThreshold = 200
    # Filter by Area.
    params.filterByArea = False
    params.minArea = 20
    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.87
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87
    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.3
    # Create a detector with the parameters
    blobdetector = cv2.SimpleBlobDetector_create(params)
    ## ===========method 1. detecting blobs================
    # #comment this out for method 2------------------------
    # # Detect blobs.
    # keypoints = blobdetector.detect(imArray_H)
    # print "length", len(keypoints)
    # # cv2.drawKeypoints(img, keypoints, img)
    # listb = []
    # for keyPoint in keypoints:
    #     bx = int(keyPoint.pt[0])
    #     by = int(keyPoint.pt[1])
    #     bs = int(keyPoint.size)
    #     print "x: ", bx, "y: ", by, "radius: ", bs
    #     cv2.circle(img, (bx,by), 1, (0, 255,255),-1)
    # ## =============method 1 part a ends============================
    # uncomment this to see part b. --------------------
    #     listb.append(list((bx,by)))
    # cnt = np.array(listb).reshape((-1, 1, 2)).astype(np.int32)
    #
    #
    # shape = (3, 3)
    # [isFound, centers] = cv2.findCirclesGrid(blank, shape, flags=cv2.CALIB_CB_ASYMMETRIC_GRID )
    #
    # print centers,isFound
    #=====method 2================================
    # #-uncomment to use---------------
    #
    # cnts = cv2.findContours(imArray_H.copy(), cv2.RETR_CCOMP,
    #                         cv2.CHAIN_APPROX_NONE)[-2]
    # # c= max(cnts[5], key=cv2.contourArea)
    # lista = []
    # for i in range(int(len(cnts))):
    #
    #     ((x, y), radius) = cv2.minEnclosingCircle(cnts[i])
    #     # print( cnts)
    #     lista.insert(i,list((x,y)))
    #     cv2.circle(img,(int(x),int(y)),int(radius),(0,255,255))
    # print lista
    #
    # cnt = np.array(lista).reshape((-1, 1, 2)).astype(np.int32)
    # # cv2.drawContours(img, [cnt], 0, (255, 255, 255), 1)
    #
    # M = cv2.moments(np.array(lista))
    # print M
    #
    # cx = int(M['m10'] / M['m00'])
    # cy = int(M['m01'] / M['m00'])
    # area = cv2.contourArea(cnt)
    #
    # r = int(np.sqrt(area/np.pi))
    # cv2.circle(img, (cx, cy), r, (0, 0, 255))
    # ellipse = cv2.fitEllipse(cnt)
    # (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
    # img = cv2.ellipse(img, ellipse, (255, 255, 0), 1)
    # print angle
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, 100)
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image,
            # cv2.circle(output, center, r, (0, 255, 0), 4)
            print "Diameter of pupil: ", 2 * r
            print "Area of Pupil : ", np.pi * r ** 2
            cv2.circle(img, (x, y), r, (0, 255, 0), 1)
        #Display result
    cv2.imshow('image',imArray_H)
    cv2.imshow('image1', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
w2d("/home/rahul/Pictures/iris1.jpeg",'db1',7)
