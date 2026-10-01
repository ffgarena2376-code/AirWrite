import cv2
import mediapipe as mp

# Open the webcam
cap = cv2.VideoCapture(0)

# Load MediaPipe Hands
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Used to draw hand landmarks
mp_draw = mp.solutions.drawing_utils

while True:

    # Capture a frame from webcam
    success, frame = cap.read()

    if not success:
        print("Failed to access webcam")
        break

    # Flip the image for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    results = hands.process(rgb_frame)

    # If a hand is detected
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Display the webcam
    cv2.imshow("AirWrite - Hand Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
cap.release()

# Close windows
cv2.destroyAllWindows()
