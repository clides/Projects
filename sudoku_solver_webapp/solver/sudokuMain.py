import os
from .utils import *
from .sudokuSolver import solve

def solve_sudoku(image):
    ######################################################### Initializing default values
    img = image
    if img.shape[2] == 3:  # If img is BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    heightImg, widthImg, c = img.shape
    model = intializePredictionModel() # load the CNN model
    status = 0
    #########################################################

    ######### 1. Preparing the Image
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
        pts2 = np.float32([[0, 0],[450, 0], [0, 450],[450, 450]]) # prepare points for warpPerspective
        matrix = cv2.getPerspectiveTransform(pts1, pts2) # compute the perspective transform
        
        imgWarpColored = cv2.warpPerspective(img, matrix, (450, 450)) # apply the perspective transform
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
            solved = solve(board)
            
            if solved:
                print("\nSolved Board:")
                print(board)
                status = 1
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
        
        imgSolvedDigits = imgBlank.copy()
        imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers)
        
        pts2 = np.float32(corners) # PREPARE POINTS FOR WARP
        pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
        imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
        
        final_solution = overlaySolutions(20, imgInvWarpColored, img)

    else:
        print("No sudoku found")


    solution = [final_solution, status]
    return solution