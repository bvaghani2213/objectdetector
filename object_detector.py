'''
Name: Brij Vaghani
B00: B00825117
Date Submitted: 28 July 2023

The main purpose of the object_detector.py script is to detect object in any given image using 
pre-trained image classification model.
To obtain this goal we will follow the following steps:
1. Apply Gaussian Filter to reduce the noise in the image or frame.
2. Apply Canny Edge Detection to Detect the edges in the image or frame.
3. Apply Adaptive Threshold to enhance the edges obtained in the Canny Edge Detector.
4. Perform Image Segmentation using kMeans Clustering to identify objects in the image.
5. Perform Image Classification to get the details of the identified object and their 
   confidence level % using pre-trained ResNet50 model.
All the implementations are explained in details in the following code.
'''
'''
Pre-Requirements: 
Run the following script in the terminal before running object_detector.py to install dependencies:
    'pip install -r requirements.txt'
'''
'''
Run object_detector:
To run object_detector run the following in terminal:
    'python3 object_detector.py <image_name>.png'
'''

# import all the required libraries and dependencies to apply the filter
import cv2
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import sys

# https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html accessed by Brij Vaghani on July 22nd 2023

def gaussian_filter(image,kernel_size):
    '''
    This method adds gausian filter to any image.
    Parameters: 
        image
        kernel_size
    Returns: 
        filtered_image
    '''
    kernel = (kernel_size,kernel_size)
    filtered_image =  cv2.GaussianBlur(image,kernel, 0)
    return filtered_image

# https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html accessed by Brij Vaghani on July 22nd 2023

def canny_edge_detector(image,lower_threshold,upper_threshold):
    '''
    This method applies canny edge detector to get all the edges present in an image.
    Parameters: 
        image
        upper_threshold
        lower_threshold
    Returns: 
        edges
    '''

    edges = cv2.Canny(image, lower_threshold, upper_threshold)
    return edges

# https://www.tutorialspoint.com/opencv/opencv_adaptive_threshold.htm accessed by Brij Vaghani on July 22nd 2023

def apply_adaptive_threshold(image,max_value = 255, C = 10 ):
    '''
    This method applies adaptive threshold to enhance the image edges using ADAPTIVE_THRESH_GAUSSIAN_C and THRESH_BINARY
    Parameters: 
        image
        max_value = 255
        C = 10
    Returns: 
        thresholded_image
    '''

    adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    threshold_type =  cv2.THRESH_BINARY
    block_size = 11
    thresholded_image = cv2.adaptiveThreshold(image, max_value, adaptive_method,threshold_type,block_size, C)

    return thresholded_image

# https://www.kdnuggets.com/2019/08/introduction-image-segmentation-k-means-clustering.html accessed by Brij Vaghani on July 25 2023.

def perform_image_segmentation(image, num_clusters):
    '''
    This method performs image segementation using KMeans on the image with enhanced edges to identify the objects.
    Parameters: 
        image
        num_clusters
    Returns: 
        segmented_image
    '''

    height, width = image.shape[:2]
    reshaped_image = image.reshape(height * width, 1)

    # initialises KMeans
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(reshaped_image)

    # Applies KMeans
    segmented_image = kmeans.cluster_centers_[kmeans.labels_].reshape(height, width)

    return segmented_image


# https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html accessed by Brij Vaghani on July 26 2023
# https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html accessed by Brij Vaghani on July 26 2023
# https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python accessed by Brij Vaghani on July 26 2023
# https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html accessed by Brij Vaghani on July 26 2023

def find_object_contours(segmented_image,original_image):
    '''
    This method find and highlight all the required objects in the image by filter out all the smaller contours using heirarchy in contour
    Parameters: 
        segemented_image
        original_image
    Returns: 
        contour_area
        rectangles_image
    '''
    # Convert Segmented image to integer for better computation
    segmented_image = segmented_image.astype(int)
    # using RETR_CCOMP hierarchy to avoid unnecessary contours.
    # it rejects all the children contours further
    hierarchy = cv2.RETR_CCOMP
    # using CHAIN_APPROX_SIMPLE appoximate method to locate contours.
    appoximation_method = cv2.CHAIN_APPROX_SIMPLE
    contours, heirarchy = cv2.findContours(segmented_image,hierarchy,appoximation_method)
    
    # variable to store each object's contour.
    objects = []
    # copy original Image to highlight objects in
    objects_image = original_image.copy()

    for i in range(len(contours)):
        # checks the heirarchy of the contour.
        # rejects the countour if the heirarchy is not parent contour.
        if heirarchy[0][i][3] != -1: 
            # rejects the contours that are considerably smaller
            if cv2.contourArea(contours[i])>25000:    
                # gets the starting point and dimenstion of the rectangle.
                object = cv2.boundingRect(contours[i])

                # checks if no two rectangles are closer than 30px
                if all(abs(object[0] - obj[0]) > 30 or abs(object[1] - obj[1]) > 30 for obj in objects):
                    # Draws the rectangle around object and appends it to the objects array
                    cv2.rectangle(objects_image, (object[0], object[1]), (object[0] + object[2], object[1] + object[3]), (0, 0, 0), 2)
                    objects.append(object)

    return objects,objects_image




# https://keras.io/api/applications/resnet/ accessed by Brij Vaghani on July 27 2023
# https://medium.com/@nutanbhogendrasharma/image-classification-with-resnet50-model-12f4c79c216b accessed by Brij Vaghani on July 27 2023
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input accessed by Brij Vaghani on July 27 2023
# https://www.geeksforgeeks.org/python-tensorflow-expand_dims/ accessed by Brij Vaghani on July 27 2023

def Resnet50_classification(objects, original_image):
    '''
    This method classifies all the objects in the image using ResNet50 Model
    Parameters: 
        objects
        original_image
    Returns: 
        final_image
    '''
    model = ResNet50(weights='imagenet', include_top=True)
    final_image = original_image.copy()

    for object in objects:
        x, y, w, h = object
        # draw rectangles to highlight object
        cv2.rectangle(final_image, (object[0], object[1]), (object[0] + object[2], object[1] + object[3]), (0, 0, 0), 2)
        # object area of the object in the image.
        object_area = final_image[y:y+h, x:x+w]
        # resize the object area to 224x224 
        object_area = cv2.resize(object_area, (224, 224))
        # Pre-Process the object area 
        object_area_preprocessed = preprocess_input(object_area)
        object_area_preprocessed = np.expand_dims(object_area_preprocessed, axis=0)

        # Perform object classification
        predictions = model.predict(object_area_preprocessed)
        predicted_label = tf.keras.applications.resnet50.decode_predictions(predictions, top=1)[0][0]
        object_class, object_description, confidence = predicted_label
        confidence_percentage = round(confidence * 100, 2)  # Calculate the confidence percentage

        # Put the classification results on top of each image
        text = f"{object_description} ({confidence_percentage}%)"
        cv2.putText(final_image, text, (x+10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return final_image





# main method to apply the filter

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Not-Found: python3 final_project.py <image.png>")
        sys.exit(1)


    # takes image path as input
    image_path = str(sys.argv[1])
    #reads the original image.
    original_image = cv2.imread(image_path)
    # converts BGR image to RGB image
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
   

    # Declaring the Kernel Size 3 for Gaussian Filter
    kernel_size = 3
    #Applying Gaussian Filter
    filtered_image = gaussian_filter(original_image,kernel_size)


    # Setting the lower threshold to 75 and upper threshold to 255 for edge detection
    lower_threshold = 75
    upper_threshold = 255
    # Apply Canny Edge Detector to perfrom Edge Detection
    edges = canny_edge_detector(filtered_image,lower_threshold, upper_threshold)


    # Apply Adaptive Threshold to the edges received from Canny Edge Detector
    thresholded_image = apply_adaptive_threshold(edges)


    # Sets the number of Cluster to perform KMeans Clustering.
    num_clusters = 6  
    # Perform Image Segmentation using KMeans Clustering
    segmented_image = perform_image_segmentation(thresholded_image, num_clusters)


    # find the object in the image using contours.
    objects_areas,object_image = find_object_contours(segmented_image,original_image)

    # performs classification in the object areas only in the image.
    final_image = Resnet50_classification(objects_areas, original_image)
    # converts BGR image to RGB image
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)


    # writes the final image as Filtere_<imagename.png>
    cv2.imwrite("Filtered_" + image_path,final_image)


