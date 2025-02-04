import cv2
import numpy as np
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from .sudokuMain import solve_sudoku

def upload_image(request):
    """
    Handles the upload of a Sudoku image and returns the same image as a base64-encoded response
    to be displayed in the frontend.

    Parameters:
    request (django.http.HttpRequest): The request object containing the uploaded image.

    Returns:
    django.http.HttpResponse: The uploaded image as a base64-encoded response.
    """

    # Check if the request method is POST and if the 'image' field is in the request FILES
    if request.method == 'POST' and 'image' in request.FILES:
        # Get the uploaded image file
        image_file = request.FILES['image']

        # Convert the uploaded file into a NumPy array and decode it
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Return the uploaded image as a base64-encoded response (to display in frontend)
        _, image_encoded = cv2.imencode('.jpg', img)
        image_bytes = image_encoded.tobytes()

        # Return the image bytes as a HttpResponse
        return HttpResponse(image_bytes, content_type="image/jpeg")

    # If the request is invalid, return the upload form
    return render(request, 'upload.html')

def solve_sudoku_image(request):
    """
    Handles the solving of a Sudoku puzzle from an uploaded image.

    Parameters:
    request (django.http.HttpRequest): The request object containing the uploaded image.

    Returns:
    django.http.HttpResponse or django.http.JsonResponse: The solution image as a JPEG response
    or a JSON response with an error message.
    """

    if request.method == "POST" and 'image' in request.FILES:
        # Read the uploaded image
        image_file = request.FILES['image']
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            # Return an error response if the image is invalid
            return JsonResponse({"error": "Invalid image file."}, status=400)

        # Call the solve_sudoku function to solve the puzzle and get the result
        solution, status = solve_sudoku(img)

        # Check the status to determine if a solution was found
        if status == 0:
            # Return an error response if no solution is found
            return JsonResponse({"status": "error", "message": "No solutions found for the given Sudoku."}, status=400)
        else:
            try:
                # Encode the solution array to an image
                _, image_encoded = cv2.imencode('.jpg', solution)
                image_bytes = image_encoded.tobytes()
                # Return the solution image as an HTTP response
                return HttpResponse(image_bytes, content_type="image/jpeg")
            except Exception as e:
                # Handle any exceptions during image encoding
                print("Error encoding image:", e)
                return JsonResponse({"error": "Failed to encode solution image."}, status=500)

    # Return an error response for invalid requests
    return JsonResponse({"error": "Invalid request"}, status=400)