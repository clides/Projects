import os
from utils import *
import sudokuSolver

######################################################### Initializing default values
pathImage = "Resources/2.jpg"
heightImg = 450
widthImg = 450
model = intializePredictionModel() # load the CNN model
#########################################################

######### 1. Preparing the Image
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg)) # resize the image to make it a square
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8) # create a blank image for testing/debugging
imgThreshold = preProcess(img)


######### 2. Find all contours
imgContours = img.copy() # copy the image for display purposes (will contain all contours)
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contours with cv2
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # draw all contours


########## 3. Finding the biggest countour
imgBigContour = img.copy() # copy the image for display purposes (will contain biggest contour)
corners, maxArea = biggestContour(contours) # find the corners of the biggest contour
print("Corners before reorder: " + str(corners))

if corners.size != 0: # if the biggest contour is found
    print("Corners after reorder: " + str(corners))
    corners = reorder(corners) # reorder the corners so it works in the format of warpPerspective
    cv2.drawContours(imgBigContour, corners, -1, (0, 0, 255), 25) # draw the biggest contour
    
    pts1 = np.float32(corners) # prepare points for warpPerspective
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # prepare points for warpPerspective
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # compute the perspective transform
    
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) # apply the perspective transform
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY) # convert to grayscale
    
    
    ######### 4. Split the image to find single digits
    boxes = splitBoxes(imgWarpColored) # split the image into 81 boxes
    print(len(boxes))
    
    # cv2.imshow("Sample",boxes[77]) # display a single box for testing
    
    numbers = getPrediction(boxes, model) # get the predictions for each box
    print(numbers)
    
    # displaying the numbers out on a blank image
    imgDetectedDigits = imgBlank.copy()
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255)) # display the numbers on the image
    
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1) # create an array of 1s (where there is not a digit) and 0s (where there is already a digit)
    print(posArray)
  
    
    ######### 5. Find solutions of the board
    board = np.array_split(numbers, 9) # split the numbers into a 9x9 grid
    
    # try to solve the board
    try:
        solved = sudokuSolver.solve(board)
        
        if solved:
            print("\nSolved Board:")
            print(board)
        else:
            print("\nBoard cannot be solved")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    # turn the board into a flat list to be able to display the numbers
    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers = flatList * posArray
    
    # display the solved numbers on the blank image
    imgSolvedDigits = imgBlank.copy()
    imgSolvedDigits = displayNumbers(imgSolvedDigits,solvedNumbers)
    
    finalImage = imgWarpColored.copy()
    finalImage = cv2.cvtColor(finalImage, cv2.COLOR_GRAY2BGR)
    finalImage = displayNumbers(finalImage, solvedNumbers)

else:
    print("No sudoku found")


# Create a image array so that we can display all the images in one window
imageArray = ([img, imgThreshold, imgContours, imgBigContour], 
              [imgWarpColored, imgDetectedDigits, imgSolvedDigits, finalImage])
stackedImage = stackImages(imageArray, 0.9)
cv2.imshow('Stacked Images', stackedImage)
cv2.imwrite("result.jpg", stackedImage)
cv2.waitKey(0)