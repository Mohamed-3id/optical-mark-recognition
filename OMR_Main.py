import cv2
import numpy as np
import Functions


########################################################################
#webCamFeed = True
pathImage = "1.jpg"
cap = cv2.VideoCapture(1)
cap.set(10,160)
heightImg = 600
widthImg  = 500
questions=10
choices=5
ans= [0,0,1,0,2,1,1,0,2,0]
########################################################################

count=0
while True:

    #if webCamFeed:success, img = cap.read()
    #else:img = cv2.imread(pathImage)
    img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg))
    imgFinal = img.copy()
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur,10,70)

    try:
        ## FIND ALL COUNTOURS
        imgContours = img.copy()
        imgBigContour = img.copy()
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 8)
        rectCon = Functions.rectContour(contours)
        biggestPoints= Functions.getCornerPoints(rectCon[0])
        gradePoints = Functions.getCornerPoints(rectCon[2])

        

        if biggestPoints.size != 0 and gradePoints.size != 0:

            # BIGGEST RECTANGLE WARPING
            biggestPoints=Functions.reorder(biggestPoints)
            cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20)
            pts1 = np.float32(biggestPoints)
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            # SECOND BIGGEST RECTANGLE WARPING
            cv2.drawContours(imgBigContour, gradePoints, -1, (255, 0, 0), 20)
            gradePoints = Functions.reorder(gradePoints)
            ptsG1 = np.float32(gradePoints)
            ptsG2 = np.float32([[0, 0], [350, 0], [0, 200], [350, 200]])
            matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (350, 200))

            # APPLY THRESHOLD
            imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray, 173, 255,cv2.THRESH_BINARY_INV )[1]

            boxes = Functions.splitBoxes(imgThresh)
            
            countR=0
            countC=0
            myPixelVal = np.zeros((questions,choices))
            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC]= totalPixels
                countC += 1
                if (countC==choices):countC=0;countR +=1
                print(myPixelVal)

            # FIND THE USER ANSWERS AND PUT THEM IN A LIST
            myIndex=[]
            for x in range (0,questions):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.amax(arr))
                myIndex.append(myIndexVal[0][0])
            print(" |( USER  ANSWERS )|",myIndex)
            print(" |( A N S W E R S )|",ans)

            # COMPARE THE VALUES TO FIND THE CORRECT ANSWERS
            grading=[]
            for x in range(0,questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else:grading.append(0)
            print(" |( G R A D I N G )|",grading)
            score = (sum(grading)/questions)*100 # FINAL GRADE
            print(" |( S  C  O  R  E )|","[",score,"]")

            # DISPLAYING ANSWERS
            Functions.showAnswers(imgWarpColored,myIndex,grading,ans)
            Functions.drawGrid(imgWarpColored)
            imgRawDrawings = np.zeros_like(imgWarpColored)
            Functions.showAnswers(imgRawDrawings, myIndex, grading, ans)
            invMatrix = cv2.getPerspectiveTransform(pts2, pts1) 
            imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg)) 

            # DISPLAY GRADE
            imgRawGrade = np.zeros_like(imgGradeDisplay,np.uint8) 
            cv2.putText(imgRawGrade,str(int(score))+"%",(70,100)
                        ,cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3) 
            invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1) 
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg)) 

            # SHOW ANSWERS AND GRADE ON FINAL IMAGE
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1,0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1,0)

            # IMAGE ARRAY FOR DISPLAY
            imageArray = ([img,imgGray,imgCanny,imgContours],
                          [imgBigContour,imgThresh,imgWarpColored,imgFinal])
            cv2.imshow("Final Result", imgFinal)
    except:
        imageArray = ([img,imgGray,imgCanny,imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    # LABELS FOR DISPLAY
    lables = [["Original","Gray","Edges","Contours"],
              ["Biggest Contour","Threshold","Warpped","Final"]]

    stackedImage = Functions.stackImages(imageArray,0.5,lables)
    cv2.imshow("Result",stackedImage)

    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg",imgFinal)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1