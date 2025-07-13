import cv2
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pyautogui

# Path to Haar Cascade
CASCADE_PATH = r"D:\codsoft\detect\haarcascade_frontalface_default.xml"

def browse_file():
    Tk().withdraw()
    file_path = askopenfilename(
        title="Select an image or video file",
        filetypes=[
            ("Media files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif *.avif *.mp4 *.avi *.mov *.mkv *.wmv"),
            ("All files", "*.*")
        ]
    )
    return file_path

def is_video_file(file_path):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    _, ext = os.path.splitext(file_path.lower())
    return ext in video_extensions

def detect_faces_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image.")
        return

    facecascade = cv2.CascadeClassifier(CASCADE_PATH)
    if facecascade.empty():
        print("Error loading cascade classifier!")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    screen_width, screen_height = pyautogui.size()
    img_resized = cv2.resize(img, (int(screen_width * 0.8), int(screen_height * 0.8)))

    cv2.imshow("Face Detection - Image", img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_faces_video(video_path):
    facecascade = cv2.CascadeClassifier(CASCADE_PATH)
    if facecascade.empty():
        print("Error loading cascade classifier!")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    screen_width, screen_height = pyautogui.size()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 2x Speed: Skip every other frame
        frame_count += 1
        if frame_count % 2 != 0:
            continue

        # Resize for performance (50%)
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Resize again to fit 80% of screen
        resized_width = int(screen_width * 0.8)
        resized_height = int(screen_height * 0.8)
        frame_resized = cv2.resize(frame, (resized_width, resized_height))

        cv2.imshow("Face Detection - Video (2x Speed)", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---- Run the program ----
selected_file = browse_file()
if selected_file:
    if not os.path.exists(CASCADE_PATH):
        print("Haar Cascade file not found!")
    elif is_video_file(selected_file):
        detect_faces_video(selected_file)
    else:
        detect_faces_image(selected_file)
else:
    print("No file selected.")
