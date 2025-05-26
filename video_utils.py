import cv2

def draw_predictions(frame, class_names, probs):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (cls, prob) in enumerate(zip(class_names, probs)):
        text = f"{cls}: {prob * 100:.2f}%"
        cv2.putText(frame, text, (10, 30 + i * 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return frame
