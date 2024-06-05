import argparse
import os
import shutil

import cv2
import numpy as np
import torch
from torchvision import transforms

from face_detection.scrfd.detector import SCRFD
from face_detection.yolov5_face.detector import Yolov5Face
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import read_features

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the face detector (Choose one of the detectors)
# detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

# Initialize the face recognizer
recognizer = iresnet_inference(
    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
)


@torch.no_grad()
def get_feature(face_image):
    """
    Extract facial features from an image using the face recognition model.

    Args:
        face_image (numpy.ndarray): Input facial image.

    Returns:
        numpy.ndarray: Extracted facial features.
    """
    # Define a series of image preprocessing steps
    face_preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Convert the image to RGB format
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Apply the defined preprocessing to the image
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)

    # Use the model to obtain facial features
    emb_img_face = recognizer(face_image)[0].cpu().numpy()

    # Normalize the features
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)
    return images_emb


def add_person_from_video(video_path, backup_dir, faces_save_dir, features_path):
    """
    Add a new person to the face recognition database from a video file.

    Args:
        video_path (str): Path to the video file.
        backup_dir (str): Directory to save backup data.
        faces_save_dir (str): Directory to save the extracted faces.
        features_path (str): Path to save face features.
    """
    # Initialize lists to store names and features of added images
    images_name = []
    images_emb = []

    # Capture video
    print(f"Attempting to open video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Read frames from the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for better detection
        scale_factor = 2.0
        frame_resized = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        # Detect faces and landmarks using the face detector
        bboxes, landmarks = detector.detect(image=frame_resized)

        # Extract faces
        for i in range(len(bboxes)):
            # Get the location of the face
            x1, y1, x2, y2, score = bboxes[i]

            # Convert coordinates back to the original scale
            x1 = int(x1 / scale_factor)
            y1 = int(y1 / scale_factor)
            x2 = int(x2 / scale_factor)
            y2 = int(y2 / scale_factor)

            # Extract the face from the image
            face_image = frame[y1:y2, x1:x2]

            # Save the face to the database
            person_face_path = os.path.join(faces_save_dir, "video_faces")
            os.makedirs(person_face_path, exist_ok=True)
            number_files = len(os.listdir(person_face_path))
            path_save_face = os.path.join(person_face_path, f"{number_files}.jpg")
            cv2.imwrite(path_save_face, face_image)

            # Extract features from the face
            images_emb.append(get_feature(face_image=face_image))
            images_name.append("video_person")

    cap.release()

    # Check if no new person is found
    if not images_emb and not images_name:
        print("No new person found!")
        return None

    # Convert lists to arrays
    images_emb = np.array(images_emb)
    images_name = np.array(images_name)

    # Read existing features if available
    features = read_features(features_path)

    if features is not None:
        # Unpack existing features
        old_images_name, old_images_emb = features

        # Combine new features with existing features
        images_name = np.hstack((old_images_name, images_name))
        images_emb = np.vstack((old_images_emb, images_emb))

        print("Update features!")

    # Save the combined features
    np.savez_compressed(features_path, images_name=images_name, images_emb=images_emb)

    # Move the video to the backup data directory
    shutil.move(video_path, backup_dir)

    print("Successfully added new person from video!")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to the video file.",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="./datasets/backup",
        help="Directory to save person data.",
    )
    parser.add_argument(
        "--faces-save-dir",
        type=str,
        default="./datasets/data/",
        help="Directory to save faces.",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="./datasets/face_features/feature",
        help="Path to save face features.",
    )
    opt = parser.parse_args()

    # Run the main function
    add_person_from_video(
        video_path=opt.video_path,
        backup_dir=opt.backup_dir,
        faces_save_dir=opt.faces_save_dir,
        features_path=opt.features_path,
    )
