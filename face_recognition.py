import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

def load_images_from_folder(folder):
    images = []
    labels = []
    for subject_dir in os.listdir(folder):
        subject_path = os.path.join(folder, subject_dir)
        for filename in os.listdir(subject_path):
            if filename.endswith(".pgm"):
                img_path = os.path.join(subject_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(subject_dir)
    return np.array(images), np.array(labels)

def preprocess_images(images):
    data = []
    for img in images:
        data.append(img.flatten())
    data = np.array(data)
    return data

def compute_eigenfaces(data, img_shape, n_components=50):
    scaler = StandardScaler().fit(data)
    scaled_data = scaler.transform(data)
    pca = PCA(n_components=n_components).fit(scaled_data)
    eigenfaces = pca.components_.reshape((n_components, img_shape[0], img_shape[1]))
    return pca, scaler, eigenfaces

def project_faces(pca, scaler, data):
    scaled_data = scaler.transform(data)
    return pca.transform(scaled_data)

def recognize_face(pca, scaler, train_data, train_labels, test_face):
    test_face = scaler.transform(test_face.reshape(1, -1))
    projected_test_face = pca.transform(test_face)
    distances = euclidean_distances(projected_test_face, train_data)
    min_distance_index = np.argmin(distances)
    sample_label = train_labels[min_distance_index]
    first_image_index = np.where(train_labels == sample_label)[0][0]
    return train_labels[first_image_index], distances[0, min_distance_index], first_image_index
