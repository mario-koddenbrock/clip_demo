import argparse
import cv2
from clip_utils import get_device, load_clip_model, preprocess_frame, compute_probabilities
from video_utils import draw_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Webcam Zero-Shot Demo (Hugging Face)")
    parser.add_argument("--classes", nargs="+", required=True, help="List of class names")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model, processor = load_clip_model()
    model.to(device)
    model.eval()

    class_names = args.classes
    print("Using classes:", class_names)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        exit()

    cv2.namedWindow("CLIP Webcam Demo", cv2.WINDOW_NORMAL)

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_pil = preprocess_frame(frame)
        probs = compute_probabilities(model, processor, image_pil, class_names, device)

        frame = draw_predictions(frame, class_names, probs)
        cv2.imshow("CLIP Webcam Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
