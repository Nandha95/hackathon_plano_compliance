import cv2
import numpy as np

def find_similar_images(query_image_path, dataset_images_paths, num_matches):
    # Load the query image
    query_image = cv2.imread(query_image_path)
    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect and compute keypoints and descriptors for the query image
    query_keypoints, query_descriptors = sift.detectAndCompute(query_gray, None)

    # Create a list to store the similarity scores
    similarity_scores = []

    # Iterate over the dataset images
    for image_path in dataset_images_paths:
        # Load the dataset image
        dataset_image = cv2.imread(image_path)
        dataset_gray = cv2.cvtColor(dataset_image, cv2.COLOR_BGR2GRAY)

        # Detect and compute keypoints and descriptors for the dataset image
        dataset_keypoints, dataset_descriptors = sift.detectAndCompute(dataset_gray, None)
        
        # Create a brute-force matcher
        matcher = cv2.BFMatcher()

        # Match the descriptors of the query and dataset images
        matches = matcher.knnMatch(query_descriptors, dataset_descriptors, k=2)

        # Apply ratio test to filter good matches
        good_matches = []
        for m,n in matches:
            if m.distance < 0.5 * n.distance:
                good_matches.append(m)

        # Compute the similarity score based on the number of good matches
        similarity_score = len(good_matches)

        # Append the similarity score and image path to the list
        similarity_scores.append((similarity_score, image_path))

    # Sort the similarity scores in descending order
    similarity_scores.sort(reverse=True)

    # Get the top similar images based on the number of matches
    top_similar_images = similarity_scores[:num_matches]

    return top_similar_images
import pandas as pd
import sys
sys.path.append(".")
from app.config import *
# Example usage
query_image_path = DETECTION_OUTPUT_PATH+'/image_0.jpg'
df = pd.read_csv(SKU_CATLOG_CSV_PATH)
dataset_images_paths = list(df['image_path'])

num_matches = 3

similar_images = find_similar_images(query_image_path, dataset_images_paths, num_matches)

# Display the top similar images and their similarity scores
for similarity_score, image_path in similar_images:
    print(f"Similarity Score: {similarity_score}, Image Path: {image_path}")
