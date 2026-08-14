import cv2
import mediapipe as mp
import time
import pyttsx3
import threading


# ============================================================
# VOICE ALERT
# ============================================================

def voice_alert():

    try:
        engine = pyttsx3.init("sapi5")

        engine.setProperty("volume", 1.0)
        engine.setProperty("rate", 130)

        voices = engine.getProperty("voices")

        if len(voices) > 0:
            engine.setProperty("voice", voices[0].id)

        engine.say("Please stay alert")
        engine.runAndWait()

        engine.stop()

    except Exception as e:

        print("Voice alert error:", e)


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),

    running_mode=RunningMode.VIDEO,

    num_faces=1
)


landmarker = FaceLandmarker.create_from_options(
    options
)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# VARIABLES
# ============================================================

timestamp = 0

blink_count = 0

eyes_closed = False

closed_start = None

closed_time = 0.0

alert_given = False


# ============================================================
# FRAME CONFIRMATION
# ============================================================

closed_frames = 0
open_frames = 0

REQUIRED_CLOSED_FRAMES = 3
REQUIRED_OPEN_FRAMES = 3


# ============================================================
# THRESHOLDS
# ============================================================

# Eye ratio below this = closed
CLOSED_THRESHOLD = 0.20

# Eye ratio above this = open
OPEN_THRESHOLD = 0.25


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = camera.read()

    if not ret:

        print("Camera could not be opened.")

        break


    # --------------------------------------------------------
    # Mirror camera
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)


    # --------------------------------------------------------
    # Convert BGR to RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # MediaPipe Image
    # --------------------------------------------------------

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp += 33


    # --------------------------------------------------------
    # Face Detection
    # --------------------------------------------------------

    result = landmarker.detect_for_video(
        image,
        timestamp
    )


    # ========================================================
    # FACE FOUND
    # ========================================================

    if result.face_landmarks:

        face = result.face_landmarks[0]


        # ----------------------------------------------------
        # LEFT EYE
        # ----------------------------------------------------

        left_vertical = abs(
            face[159].y - face[145].y
        )

        left_horizontal = abs(
            face[33].x - face[133].x
        )


        # ----------------------------------------------------
        # RIGHT EYE
        # ----------------------------------------------------

        right_vertical = abs(
            face[386].y - face[374].y
        )

        right_horizontal = abs(
            face[362].x - face[263].x
        )


        # ----------------------------------------------------
        # EYE RATIO
        # ----------------------------------------------------

        if (
            left_horizontal > 0
            and right_horizontal > 0
        ):

            left_ratio = (
                left_vertical /
                left_horizontal
            )

            right_ratio = (
                right_vertical /
                right_horizontal
            )

            eye_ratio = (
                left_ratio +
                right_ratio
            ) / 2

        else:

            eye_ratio = 0.30


        # ====================================================
        # EYE STATE DETECTION
        # ====================================================

        # ----------------------------------------------------
        # POSSIBLY CLOSED
        # ----------------------------------------------------

        if eye_ratio < CLOSED_THRESHOLD:

            closed_frames += 1
            open_frames = 0


            # Start closed state only after
            # several consecutive closed frames

            if (
                not eyes_closed
                and closed_frames >= REQUIRED_CLOSED_FRAMES
            ):

                eyes_closed = True

                closed_start = time.monotonic()

                closed_time = 0.0

                alert_given = False


        # ----------------------------------------------------
        # POSSIBLY OPEN
        # ----------------------------------------------------

        elif eye_ratio > OPEN_THRESHOLD:

            open_frames += 1
            closed_frames = 0


            # Confirm eyes are really open

            if (
                eyes_closed
                and open_frames >= REQUIRED_OPEN_FRAMES
            ):

                # Get final closed duration

                closed_time = (
                    time.monotonic()
                    - closed_start
                )


                # ------------------------------------------------
                # BLINK COUNT
                # ------------------------------------------------

                if (
                    closed_time >= 0.08
                    and closed_time < 6.0
                ):

                    blink_count += 1


                # Reset

                eyes_closed = False

                closed_start = None

                closed_time = 0.0

                alert_given = False


        # ----------------------------------------------------
        # BETWEEN THRESHOLDS
        # ----------------------------------------------------

        else:

            # Do NOT reset the state here.
            #
            # This is important.
            #
            # Small eye movements will not reset
            # the 6-second timer.

            pass


        # ====================================================
        # CALCULATE CLOSED TIME
        # ====================================================

        if eyes_closed and closed_start is not None:

            closed_time = (
                time.monotonic()
                - closed_start
            )


        # ====================================================
        # SIX SECOND ALERT
        # ====================================================

        if (
            eyes_closed
            and closed_time >= 6.0
            and not alert_given
        ):

            alert_given = True


            # Display alert immediately

            status = "PLEASE STAY ALERT"


            # Start voice alert

            alert_thread = threading.Thread(
                target=voice_alert,
                daemon=True
            )

            alert_thread.start()


        elif eyes_closed:

            status = "Eyes Closed"


        else:

            status = "Normal"


        # ====================================================
        # EYE STATUS
        # ====================================================

        if eyes_closed:

            eye_text = "Eyes Closed"

        else:

            eye_text = "Eyes Open"


        cv2.putText(

            frame,

            eye_text,

            (40, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 0),

            2
        )


        # ====================================================
        # BLINK COUNT
        # ====================================================

        cv2.putText(

            frame,

            "Blink Count: " +
            str(blink_count),

            (40, 90),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 255),

            2
        )


        # ====================================================
        # CLOSED TIME
        # ====================================================

        cv2.putText(

            frame,

            "Closed Time: {:.1f}s".format(
                closed_time
            ),

            (40, 130),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2
        )


        # ====================================================
        # EYE RATIO
        # ====================================================

        cv2.putText(

            frame,

            "Eye Ratio: {:.2f}".format(
                eye_ratio
            ),

            (40, 165),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (200, 200, 200),

            2
        )


        # ====================================================
        # STATUS
        # ====================================================

        cv2.putText(

            frame,

            status,

            (40, 205),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 0, 255),

            2
        )


    # ========================================================
    # NO FACE
    # ========================================================

    else:

        cv2.putText(

            frame,

            "No Face Detected",

            (40, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 0, 255),

            2
        )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "Real-Time Eye Detection",
        frame
    )


    # ========================================================
    # PRESS Q TO EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# RELEASE RESOURCES
# ============================================================

camera.release()

cv2.destroyAllWindows()

landmarker.close()