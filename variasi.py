import cv2 as cv
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def getHandMove(hand_landmarks):
    landmarks = hand_landmarks.landmark
    if all([landmarks[i].y < landmarks[i+3].y for i in range(9,20,4)]): 
        return "rock"
    elif landmarks[13].y < landmarks[16].y and landmarks[17].y < landmarks[20].y: 
        return "scissors"
    else: 
        return "paper"

# Setel resolusi atau ukuran jendela tampilan
frame_width = 1280  # Ganti dengan lebar yang diinginkan
frame_height = 720  # Ganti dengan tinggi yang diinginkan

vid = cv.VideoCapture(0)
vid.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

clock = 0
p1_move = p2_move = None
gameText = ""
success = True

with mp_hands.Hands(model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = vid.read()
        if not ret or frame is None:
            continue

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = hands.process(frame)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame,
                                          hand_landmarks,
                                          mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())
                bbox_top_left = (int(min(p.x for p in hand_landmarks.landmark) * frame.shape[1]),
                                int(min(p.y for p in hand_landmarks.landmark) * frame.shape[0]))
                bbox_bottom_right = (int(max(p.x for p in hand_landmarks.landmark) * frame.shape[1]),
                                     int(max(p.y for p in hand_landmarks.landmark) * frame.shape[0]))
                cv.rectangle(frame, bbox_top_left, bbox_bottom_right, (0, 255, 0), 2)
                cv.putText(frame, f"Player {idx + 1}", bbox_top_left, cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2, cv.LINE_AA)

        frame = cv.flip(frame, 1)

        if 0 <= clock < 20:
            success = True
            gameText = "Ready?"
        elif clock < 30: gameText = "3..."
        elif clock < 40: gameText = "2..."
        elif clock < 50: gameText = "1..."
        elif clock < 60: gameText = "GO!!!"
        elif clock == 60:
            hls = result.multi_hand_landmarks
            if hls and len(hls) == 2:
                p1_move = getHandMove(hls[0])
                p2_move = getHandMove(hls[1])
            else:
                success = False
        elif clock < 100:
            if success:
                gameText = f"Played 1 played {p1_move}. Player 2 played {p2_move}."
                if p1_move == p2_move: gameText = f"{gameText} Game is tied."
                elif p1_move == "paper" and p2_move == "rock": gameText = f"{gameText} Player 1 wins."
                elif p1_move == "rock" and p2_move == "scissors": gameText = f"{gameText} Player 1 wins."
                elif p1_move == "scissors" and p2_move == "paper": gameText = f"{gameText} Player 1 wins."
                else: gameText = f"{gameText} Player 2 wins."
            else:
                gameText = "Didn't play properly!"

        cv.putText(frame, f"Clock: {clock}", (50, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv.LINE_AA)
        cv.putText(frame, gameText, (50, 80), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv.LINE_AA)

        clock = (clock + 1) % 100

        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

vid.release()
cv.destroyAllWindows()
