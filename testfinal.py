import cv2
import pyautogui
import time
import mediapipe as mp
import google.generativeai as genai
import threading

# === Configure Gemini API ===
genai.configure(api_key="your api key")
model = genai.GenerativeModel("gemini-2.0-flash")

# === Initialize MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# === Keyboard Layout ===
keys = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
    ['Shift', 'Space', 'Back', 'Correct']
]
key_width, key_height = 45, 50
special_width = 90
start_y, spacing, row_spacing = 140, 3, 6

# === Build Button Layout ===
def build_buttons():
    layout = []
    for row_idx, row in enumerate(keys):
        row_total_width = sum(special_width if key in ['Shift', 'Space', 'Back', 'Correct'] else key_width for key in row) + spacing * (len(row) - 1)
        offset_x = (640 - row_total_width) // 2
        for key in row:
            w = special_width if key in ['Shift', 'Space', 'Back', 'Correct'] else key_width
            layout.append([key, offset_x, start_y + row_idx * (key_height + row_spacing), w, key_height])
            offset_x += w + spacing
    return layout

button_layout = build_buttons()

# === State Variables ===
shift = False
typed_text = ""
last_click_time = 0
click_delay = 0.5
last_key_pressed = None
correction_done_text = ""
last_tap_time = {}
double_tap_interval = 0.5  # Interval (in seconds) within which a double-tap is recognized

# === Drawing Utilities ===
def draw_button(frame, key, x, y, w, h, active=False):
    color = (0, 255, 255) if active else (0, 0, 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x + 10, y), (x + w - 10, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + 10), (x + w, y + h - 10), color, -1)
    cv2.circle(overlay, (x + 10, y + 10), 10, color, -1)
    cv2.circle(overlay, (x + w - 10, y + 10), 10, color, -1)
    cv2.circle(overlay, (x + 10, y + h - 10), 10, color, -1)
    cv2.circle(overlay, (x + w - 10, y + h - 10), 10, color, -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    label = key.upper() if (shift and key.isalpha()) else key
    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(frame, label, (x + (w - size[0]) // 2, y + (h + size[1]) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

def draw_keyboard(frame):
    for key, x, y, w, h in button_layout:
        frame = draw_button(frame, key, x, y, w, h)
    return frame

# === Gemini Correction ===
def correct_last_word(text):
    words = text.strip().split()
    if not words:
        return text
    last = words[-1]
    try:
        prompt = f"Correct the spelling of the word: '{last}'. Only return the corrected word."
        result = model.generate_content(prompt).text.strip()
        if result and result.lower() != last.lower():
            words[-1] = result
        return " ".join(words) + " "
    except Exception as e:
        print("Gemini Error:", e)
        return text

# === Virtual Typing Actions ===
def handle_keypress(key):
    global shift, typed_text, correction_done_text

    if key == 'Shift':
        shift = not shift
    elif key == 'Space':
        pyautogui.press('space')
        typed_text += ' '
    elif key == 'Back':
        pyautogui.press('backspace')
        typed_text = typed_text[:-1]
    elif key == 'Correct':
        def correct():
            global typed_text, correction_done_text
            words = typed_text.strip().split()
            if words:
                wrong = words[-1]
                typed_text = correct_last_word(typed_text)
                corrected = typed_text.strip().split()[-1]
                for _ in range(len(wrong)):
                    pyautogui.press('backspace')
                pyautogui.write(corrected, interval=0.05)
                correction_done_text = typed_text
        threading.Thread(target=correct).start()
    else:
        char = key.upper() if shift else key.lower()
        pyautogui.write(char)
        typed_text += char
        shift = False

# === Main Camera Loop ===
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(5, 15)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    frame = draw_keyboard(frame)
    cv2.putText(frame, f"Typing: {typed_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    pressed_this_frame = None

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            lm = [(int(p.x * w), int(p.y * h)) for p in hand.landmark]
            index, middle = lm[8], lm[12]
            dist = ((index[0]-middle[0])**2 + (index[1]-middle[1])**2)**0.5

            if dist < 40:
                for key, x, y, w_, h_ in button_layout:
                    if x < index[0] < x + w_ and y < index[1] < y + h_:
                        frame = draw_button(frame, key, x, y, w_, h_, active=True)
                        
                        # Check if the button was tapped twice
                        current_time = time.time()
                        if key not in last_tap_time:
                            last_tap_time[key] = current_time
                        else:
                            time_diff = current_time - last_tap_time[key]
                            if time_diff < double_tap_interval:  # Double-tap detected
                                if key != last_key_pressed and time.time() - last_click_time > click_delay:
                                    last_click_time = time.time()
                                    handle_keypress(key)
                                    last_key_pressed = key
                            last_tap_time[key] = current_time
                        pressed_this_frame = key

    if not pressed_this_frame:
        last_key_pressed = None

    cv2.imshow("Virtual Keyboard", frame)
    if cv2.waitKey(1) in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
