# ============================================================
# AIRWRITE: GESTURE-BASED TEXT RECOGNITION
# Module 1
# Hand Detection + Fingertip Tracking + Air Writing
# ============================================================

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# 1. SETTINGS
# ============================================================

CAMERA_INDEX = 0

MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

MAX_JUMP_DISTANCE = 100

LINE_THICKNESS = 5


# ============================================================
# 2. OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    print("Make sure a webcam is available.")
    exit()


# ============================================================
# 3. MEDIAPIPE HANDS
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)


# ============================================================
# 4. VARIABLES
# ============================================================

points = []

previous_point = None

frame_count = 0

detected_frames = 0


# ============================================================
# 5. START AIRWRITE
# ============================================================

print("==========================================")
print("       AIRWRITE: GESTURE TEXT")
print("==========================================")
print("Camera started successfully.")
print()
print("Move your INDEX FINGER to write in the air.")
print()
print("Press C = Clear drawing")
print("Press Q = Quit")
print("==========================================")


while True:

    # --------------------------------------------------------
    # Capture frame
    # --------------------------------------------------------

    success, frame = cap.read()

    if not success:

        print("ERROR: Could not read camera frame.")
        break

    frame_count += 1


    # --------------------------------------------------------
    # Mirror camera
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)


    # --------------------------------------------------------
    # Get frame dimensions
    # --------------------------------------------------------

    height, width, _ = frame.shape


    # --------------------------------------------------------
    # Convert BGR to RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Detect hand
    # --------------------------------------------------------

    results = hands.process(rgb_frame)


    # ========================================================
    # 6. HAND DETECTED
    # ========================================================

    if results.multi_hand_landmarks:

        detected_frames += 1

        hand_landmarks = results.multi_hand_landmarks[0]


        # ----------------------------------------------------
        # Draw 21 hand landmarks
        # ----------------------------------------------------

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )


        # ----------------------------------------------------
        # Index fingertip
        #
        # MediaPipe landmark 8 = index fingertip
        # ----------------------------------------------------

        index_tip = hand_landmarks.landmark[8]


        # ----------------------------------------------------
        # Convert normalized coordinates to pixels
        # ----------------------------------------------------

        x = int(index_tip.x * width)
        y = int(index_tip.y * height)


        # ----------------------------------------------------
        # Draw fingertip marker
        # ----------------------------------------------------

        cv2.circle(
            frame,
            (x, y),
            10,
            (0, 255, 0),
            -1
        )


        # ----------------------------------------------------
        # Store fingertip movement
        # ----------------------------------------------------

        current_point = (x, y)


        if previous_point is not None:

            old_x, old_y = previous_point

            distance = np.sqrt(
                (x - old_x) ** 2 +
                (y - old_y) ** 2
            )


            # Prevent sudden tracking jumps

            if distance < MAX_JUMP_DISTANCE:

                points.append(current_point)


                # Draw air-writing line

                cv2.line(
                    frame,
                    previous_point,
                    current_point,
                    (255, 0, 0),
                    LINE_THICKNESS
                )


        else:

            points.append(current_point)


        previous_point = current_point


    # ========================================================
    # 7. HAND NOT DETECTED
    # ========================================================

    else:

        # Stop connecting the next stroke
        previous_point = None


    # ========================================================
    # 8. DISPLAY INFORMATION
    # ========================================================

    cv2.putText(
        frame,
        "AIRWRITE",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "Index Finger Air Writing",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "C: Clear   Q: Quit",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # 9. SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "AirWrite - Gesture Based Text Recognition",
        frame
    )


    # ========================================================
    # 10. KEYBOARD CONTROLS
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # Clear drawing

    if key == ord("c"):

        points.clear()

        previous_point = None

        print("Drawing cleared.")


    # Quit

    elif key == ord("q"):

        break


# ============================================================
# 11. CLEAN UP
# ============================================================

cap.release()

hands.close()

cv2.destroyAllWindows()


# ============================================================
# 12. FINAL INFORMATION
# ============================================================

print()
print("==========================================")
print("          AIRWRITE FINISHED")
print("==========================================")
print("Frames processed:", frame_count)
print("Hand detected frames:", detected_frames)
print("Fingertip points:", len(points))
print("==========================================")
