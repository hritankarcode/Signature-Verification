


import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def compare_images_with_kmeans_and_orb(img1, img2, kmeans_threshold=30, orb_threshold=100):
    # Step 1: K-Means for Color Comparison
    def kmeans_similarity(image1, image2):
        # Convert to HSV
        img1_hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
        img2_hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

        # Flatten images
        img1_pixels = img1_hsv.reshape((-1, 3))
        img2_pixels = img2_hsv.reshape((-1, 3))

        # KMeans clustering
        kmeans1 = KMeans(n_clusters=3, random_state=42)
        kmeans2 = KMeans(n_clusters=3, random_state=42)

        kmeans1.fit(img1_pixels)
        kmeans2.fit(img2_pixels)

        # Compare cluster centers
        centers1 = kmeans1.cluster_centers_
        centers2 = kmeans2.cluster_centers_

        # Compute average distance
        distances = np.linalg.norm(centers1 - centers2, axis=1)
        return np.mean(distances)

    # Step 2: ORB for Structural Comparison
    def orb_similarity(image1, image2):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(image1, None)
        kp2, des2 = orb.detectAndCompute(image2, None)

        if des1 is None or des2 is None:
            return 0  # No matches if descriptors are missing

        # BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Visualize matches
        img_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(16, 16))
        plt.title("ORB Matches")
        plt.imshow(img_matches[..., ::-1])  # Convert BGR to RGB for display
        plt.axis('off')
        plt.show(block=False)  # Non-blocking show
        plt.pause(5)  # Keep window open for 5 seconds
        plt.close()  # Close the window to resume execution

        # Return number of good matches
        return len(matches)

    # Step 3: Perform K-Means and ORB Comparisons
    kmeans_dist = kmeans_similarity(img1, img2)
    orb_matches = orb_similarity(img1, img2)

    # Step 4: Make Decision Based on Thresholds
    print(f"K-Means Distance: {kmeans_dist}")
    print(f"ORB Matches: {orb_matches}")

    if kmeans_dist < kmeans_threshold and orb_matches > orb_threshold:
        print("Images are similar!")
    else:
        print("Images are not similar!")

# Load Images
  # Train Image

img1 = cv2.imread("images/image1.jpeg")
img2 = cv2.imread("images/image2.jpeg")

# Display Input Images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Image 1")
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Image 2")
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Run Comparison
compare_images_with_kmeans_and_orb(img1, img2)

