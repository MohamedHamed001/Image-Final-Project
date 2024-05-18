from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QLineEdit, QFileDialog, QSizePolicy, QGridLayout, QMessageBox, QWidget, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
from collections import defaultdict
from face_recognition import load_images_from_folder, preprocess_images, compute_eigenfaces, project_faces, recognize_face

def get_edges(img, min_edge_threshold=100, max_edge_threshold=200):
    min_edge_threshold = 0.2 * np.max(img)
    max_edge_threshold = 0.6 * np.max(img)
    edge_image = cv2.Canny(img, min_edge_threshold, max_edge_threshold)
    return edge_image

def precompute_trig(theta_range):
    cos_theta = {}
    sin_theta = {}
    for theta in theta_range:
        cos_theta[theta] = np.cos(theta)
        sin_theta[theta] = np.sin(theta)
    return cos_theta, sin_theta

def hough_ellipse_visual(image, a_range, b_range, theta_range):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = get_edges(blurred)
    
    cos_theta, sin_theta = precompute_trig(theta_range)
    accumulator = defaultdict(int)
    
    for y, x in np.argwhere(edges):
        for a in range(*a_range):
            for b in range(*b_range):
                for theta in theta_range:
                    x_c = int(round(x - a * cos_theta[theta]))
                    y_c = int(round(y + b * sin_theta[theta]))
                    if 0 <= x_c < image.shape[1] and 0 <= y_c < image.shape[0]:
                        accumulator[(y_c, x_c, a, b, theta)] += 1

    max_accumulator = max(accumulator.values())
    threshold = max_accumulator * 0.5
    potential_ellipses = [k for k, v in accumulator.items() if v >= threshold]
    
    for y_c, x_c, a, b, theta in potential_ellipses:
        scaled_a = int(a * 0.1)
        scaled_b = int(b * 0.1)
        cv2.ellipse(image, (x_c, y_c), (scaled_a, scaled_b), np.degrees(theta), 0, 360, (0, 255, 0), 1)
    
    return image


class HoughEllipseWorker(QThread):
    result_ready = pyqtSignal(np.ndarray)

    def __init__(self, image, a_range, b_range, theta_range):
        super().__init__()
        self.image = image
        self.a_range = a_range
        self.b_range = b_range
        self.theta_range = theta_range

    def run(self):
        result_image = hough_ellipse_visual(self.image, self.a_range, self.b_range, self.theta_range)
        self.result_ready.emit(result_image)

class FaceRecognitionWorker(QThread):
    result_ready = pyqtSignal(np.ndarray, str, np.ndarray)

    def __init__(self, image, pca, scaler, projected_faces, labels, img_shape, images):
        super().__init__()
        self.image = image
        self.pca = pca
        self.scaler = scaler
        self.projected_faces = projected_faces
        self.labels = labels
        self.img_shape = img_shape
        self.images = images

    def run(self):
        resized_image = cv2.resize(self.image, (self.img_shape[1], self.img_shape[0]))
        predicted_label, distance, first_image_index = recognize_face(self.pca, self.scaler, self.projected_faces, self.labels, resized_image.flatten())
        closest_match_image = self.images[first_image_index]
        self.result_ready.emit(closest_match_image, predicted_label, resized_image)

class HoughEllipseDetectionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Processing - Hough Ellipse Detection and Face Recognition')
        self.setFixedSize(800, 600)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)
        self.fixed_size = (350, 350)  # Fixed size for all images
        self.init_ui()
        self.loaded_image = None
        self.worker = None

        # Load face recognition data
        self.folder = r'dataset'  # Update this to the correct path
        self.images, self.labels = load_images_from_folder(self.folder)
        self.img_shape = self.images[0].shape
        self.data = preprocess_images(self.images)
        self.n_components = 50
        self.pca, self.scaler, self.eigenfaces = compute_eigenfaces(self.data, self.img_shape, n_components=self.n_components)
        self.projected_faces = project_faces(self.pca, self.scaler, self.data)

    def init_ui(self):
        self.tabs = QTabWidget()
        self.hough_tab = QWidget()
        self.face_recognition_tab = QWidget()

        self.tabs.addTab(self.hough_tab, "Hough Ellipse Detection")
        self.tabs.addTab(self.face_recognition_tab, "Face Recognition")

        self.init_hough_tab()
        self.init_face_recognition_tab()

        self.setCentralWidget(self.tabs)

    def init_hough_tab(self):
        self.hough_layout = QVBoxLayout()
        
        self.hough_top_layout = QHBoxLayout()
        self.browse_button = QPushButton('Browse', self)
        self.apply_button = QPushButton('Apply', self)
        self.hough_top_layout.addWidget(self.browse_button)
        self.hough_top_layout.addWidget(self.apply_button)
        
        self.hough_images_layout = QHBoxLayout()
        self.original_image_layout = QVBoxLayout()
        
        self.original_label = QLabel('Original')
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet('font-weight: bold; font-size: 16px;')
        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setFixedSize(*self.fixed_size)
        self.original_image_label.setStyleSheet("background-color: black")
        self.original_image_layout.addWidget(self.original_label)
        self.original_image_layout.addWidget(self.original_image_label)
        
        self.detected_image_layout = QVBoxLayout()
        self.detected_label = QLabel('Result')
        self.detected_label.setAlignment(Qt.AlignCenter)
        self.detected_label.setStyleSheet('font-weight: bold; font-size: 16px;')
        self.detected_lines_image_label = QLabel()
        self.detected_lines_image_label.setAlignment(Qt.AlignCenter)
        self.detected_lines_image_label.setFixedSize(*self.fixed_size)
        self.detected_lines_image_label.setStyleSheet("background-color: white")
        self.detected_image_layout.addWidget(self.detected_label)
        self.detected_image_layout.addWidget(self.detected_lines_image_label)
        
        self.hough_images_layout.addLayout(self.original_image_layout)
        self.hough_images_layout.addLayout(self.detected_image_layout)

        self.hough_controls_layout = QGridLayout()
        
        self.a_range_input = QLineEdit()
        self.hough_controls_layout.addWidget(QLabel('a Range (min,max)'), 1, 0)
        self.hough_controls_layout.addWidget(self.a_range_input, 2, 0)

        self.b_range_input = QLineEdit()
        self.hough_controls_layout.addWidget(QLabel('b Range (min,max)'), 1, 1)
        self.hough_controls_layout.addWidget(self.b_range_input, 2, 1)

        self.theta_range_input = QLineEdit()
        self.hough_controls_layout.addWidget(QLabel('Theta Range (degrees)'), 1, 2)
        self.hough_controls_layout.addWidget(self.theta_range_input, 2, 2)

        self.hough_status_label = QLabel('Ready')
        self.hough_status_label.setAlignment(Qt.AlignCenter)
        self.hough_status_label.setStyleSheet('font-weight: bold; font-size: 14px; color: green;')

        self.hough_layout.addLayout(self.hough_top_layout)
        self.hough_layout.addLayout(self.hough_images_layout)
        self.hough_layout.addLayout(self.hough_controls_layout)
        self.hough_layout.addWidget(self.hough_status_label)

        self.browse_button.clicked.connect(self.browse_image)
        self.apply_button.clicked.connect(self.apply_hough_ellipse_transform)
        
        self.hough_tab.setLayout(self.hough_layout)

    def init_face_recognition_tab(self):
        self.face_layout = QVBoxLayout()

        self.face_top_layout = QHBoxLayout()
        self.face_browse_button = QPushButton('Browse', self)
        self.face_recognize_button = QPushButton('Detect', self)
        self.face_top_layout.addWidget(self.face_browse_button)
        self.face_top_layout.addWidget(self.face_recognize_button)

        self.face_images_layout = QHBoxLayout()
        self.face_original_image_layout = QVBoxLayout()

        self.face_original_label = QLabel('Test Image')
        self.face_original_label.setAlignment(Qt.AlignCenter)
        self.face_original_label.setStyleSheet('font-weight: bold; font-size: 16px;')
        self.face_original_image_label = QLabel()
        self.face_original_image_label.setAlignment(Qt.AlignCenter)
        self.face_original_image_label.setFixedSize(*self.fixed_size)
        self.face_original_image_label.setStyleSheet("background-color: black")
        self.face_original_image_layout.addWidget(self.face_original_label)
        self.face_original_image_layout.addWidget(self.face_original_image_label)

        self.face_detected_image_layout = QVBoxLayout()
        self.face_detected_label = QLabel('Closest Match')
        self.face_detected_label.setAlignment(Qt.AlignCenter)
        self.face_detected_label.setStyleSheet('font-weight: bold; font-size: 16px;')
        self.face_detected_lines_image_label = QLabel()
        self.face_detected_lines_image_label.setAlignment(Qt.AlignCenter)
        self.face_detected_lines_image_label.setFixedSize(*self.fixed_size)
        self.face_detected_lines_image_label.setStyleSheet("background-color: white")
        self.face_detected_image_layout.addWidget(self.face_detected_label)
        self.face_detected_image_layout.addWidget(self.face_detected_lines_image_label)

        self.face_images_layout.addLayout(self.face_original_image_layout)
        self.face_images_layout.addLayout(self.face_detected_image_layout)

        self.face_status_label = QLabel('Ready')
        self.face_status_label.setAlignment(Qt.AlignCenter)
        self.face_status_label.setStyleSheet('font-weight: bold; font-size: 14px; color: green;')

        self.face_layout.addLayout(self.face_top_layout)
        self.face_layout.addLayout(self.face_images_layout)
        self.face_layout.addWidget(self.face_status_label)

        self.face_browse_button.clicked.connect(self.browse_face_image)
        self.face_recognize_button.clicked.connect(self.apply_face_recognition)

        self.face_recognition_tab.setLayout(self.face_layout)

    def browse_image(self):
        if hasattr(self, '_is_browsing') and self._is_browsing:
            return
        
        self._is_browsing = True
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        self._is_browsing = False

        if filename:
            self.load_image_to_label(filename, self.original_image_label, color=True)

    def browse_face_image(self):
        if hasattr(self, '_is_browsing') and self._is_browsing:
            return
        
        self._is_browsing = True
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.pgm)")
        self._is_browsing = False

        if filename:
            self.load_image_to_label(filename, self.face_original_image_label, color=False)

    def load_image_to_label(self, image_path, label, color=True):
        if color:
            image = cv2.imread(image_path)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            QMessageBox.information(self, "Error", "Failed to load image. Check the file path and format.")
            return

        self.loaded_image = image
        self.display_image(image, label)

    def display_image(self, image, label):
        if image.ndim == 3:  # Color image
            qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888).rgbSwapped()
        else:  # Grayscale image
            qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.fixed_size[0], self.fixed_size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
        label.setFixedSize(scaled_pixmap.size())

    def apply_hough_ellipse_transform(self):
        if self.loaded_image is None:
            QMessageBox.information(self, "Error", "No image loaded!")
            return

        try:
            a_range = tuple(map(int, self.a_range_input.text().split(','))) if self.a_range_input.text() else (10, 20)
            b_range = tuple(map(int, self.b_range_input.text().split(','))) if self.b_range_input.text() else (10, 20)
            theta_step = int(self.theta_range_input.text()) if self.theta_range_input.text() else 1
            theta_range = np.deg2rad(np.arange(0, 90, theta_step))
        except Exception as e:
            QMessageBox.information(self, "Error", "Invalid input values!")
            return

        self.hough_status_label.setText('Processing...')
        self.hough_status_label.setStyleSheet('font-weight: bold; font-size: 14px; color: orange;')

        self.worker = HoughEllipseWorker(self.loaded_image.copy(), a_range, b_range, theta_range)
        self.worker.result_ready.connect(self.display_result)
        self.worker.finished.connect(self.processing_finished)
        self.worker.start()

    def apply_face_recognition(self):
        if self.loaded_image is None:
            QMessageBox.information(self, "Error", "No image loaded!")
            return

        self.face_status_label.setText('Processing...')
        self.face_status_label.setStyleSheet('font-weight: bold; font-size: 14px; color: orange;')

        self.worker = FaceRecognitionWorker(self.loaded_image.copy(), self.pca, self.scaler, self.projected_faces, self.labels, self.img_shape, self.images)
        self.worker.result_ready.connect(self.display_face_result)
        self.worker.finished.connect(self.face_recognition_finished)
        self.worker.start()

    def display_result(self, result_image):
        self.display_image(result_image, self.detected_lines_image_label)

    def display_face_result(self, closest_match_image, predicted_label, test_image):
        self.display_image(closest_match_image, self.face_detected_lines_image_label)
        QMessageBox.information(self, "Result", f"Recognized as: {predicted_label}")

    def processing_finished(self):
        self.hough_status_label.setText('Processing Finished')
        self.hough_status_label.setStyleSheet('font-weight: bold; font-size: 14px; color: green;')

    def face_recognition_finished(self):
        self.face_status_label.setText('Processing Finished')
        self.face_status_label.setStyleSheet('font-weight: bold; font-size: 14px; color: green;')

if __name__ == '__main__':
    app = QApplication([])
    window = HoughEllipseDetectionUI()
    window.show()
    app.exec_()
