import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Load images from a directory and convert them to grayscale
def load_images_from_folder(folder):
    images = []
    labels = []
    for subject_dir in os.listdir(folder):
        subject_path = os.path.join(folder, subject_dir)
        for filename in os.listdir(subject_path):
            img_path = os.path.join(subject_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(subject_dir)
    return np.array(images), np.array(labels)

# Reshape and standardize the image data
def preprocess_images(images):
    data = []
    for img in images:
        data.append(img.flatten())
    data = np.array(data)
    return data

# Perform PCA to obtain eigenfaces
def compute_eigenfaces(data, n_components=50):
    scaler = StandardScaler().fit(data)
    scaled_data = scaler.transform(data)
    pca = PCA(n_components=n_components).fit(scaled_data)
    eigenfaces = pca.components_.reshape((n_components, img_shape[0], img_shape[1]))
    return pca, scaler, eigenfaces

# Project the face images onto the eigenface space
def project_faces(pca, scaler, data):
    scaled_data = scaler.transform(data)
    return pca.transform(scaled_data)

# Recognize a new face by finding the closest face in the training set
def recognize_face(pca, scaler, train_data, train_labels, test_face):
    test_face = scaler.transform(test_face.reshape(1, -1))
    projected_test_face = pca.transform(test_face)
    distances = euclidean_distances(projected_test_face, train_data)
    min_distance_index = np.argmin(distances)
    return train_labels[min_distance_index], distances[0, min_distance_index]

def choose_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        test_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if test_image is not None:
            # Resize test image to match training images
            test_image = cv2.resize(test_image, (img_shape[1], img_shape[0]))
            # Recognize the test image
            predicted_label, distance = recognize_face(pca, scaler, projected_faces, labels, test_image.flatten())
            print(f'Recognized as: {predicted_label} with distance: {distance}')
            display_results(test_image, predicted_label)

def display_results(test_image, predicted_label):
    plt.subplot(1, 2, 1)
    plt.title('Test Image')
    plt.imshow(test_image, cmap='gray')
    
    closest_match_index = np.where(labels == predicted_label)[0][0]
    closest_match_image = images[closest_match_index]
    
    plt.subplot(1, 2, 2)
    plt.title('Closest Match')
    plt.imshow(closest_match_image, cmap='gray')
    
    plt.show()

# Load the dataset
folder = r'dataset'
images, labels = load_images_from_folder(folder)

# Preprocess the images
img_shape = images[0].shape
data = preprocess_images(images)

# Compute eigenfaces
n_components = 50
pca, scaler, eigenfaces = compute_eigenfaces(data, n_components=n_components)

# Project the face images onto the eigenface space
projected_faces = project_faces(pca, scaler, data)

# Create a GUI window
root = tk.Tk()
root.title("Face Recognition using Eigenfaces")
root.geometry("800x600")  # Set the initial size of the window to 800x600 pixels

# Add a button to choose an image
button = tk.Button(root, text="Choose Image", command=choose_image)
button.pack()

# Start the GUI event loop
root.mainloop()
