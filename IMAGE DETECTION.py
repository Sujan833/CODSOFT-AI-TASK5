import cv2
import os

def detect_faces_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image.")
        return
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade_path = r"D:\codsoft\detect\haarcascade_frontalface_default.xml"
    
    if not os.path.exists(cascade_path):
        print("Haar Cascade file not found!")
        return

    facecascade = cv2.CascadeClassifier(cascade_path)
    if facecascade.empty():
        print("Error loading cascade classifier!")
        return

    faces = facecascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected.")
    else:
        print(f"Found {len(faces)} face(s).")

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Faces detected", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_faces_image(r"D:\codsoft\detect\images.jpeg")
detect_faces_image(r"D:\codsoft\detect\images (1).jpeg")
