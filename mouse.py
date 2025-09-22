import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np
from collections import deque
from pygetwindow import getWindowsWithTitle, getActiveWindow

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Thresholds (adjust as needed)
CLICK_THRESHOLD = 0.05
SWIPE_THRESHOLD = 0.15
SWIPE_MIN_DISTANCE = 0.3
WINDOW_SWAP_ACTIVATION_DISTANCE = 0.4
DEBOUNCE_TIME = 0.5  # Minimum time between window switches

# Cursor smoothing parameters
SMOOTHING_HISTORY_LENGTH = 5
cursor_history = deque(maxlen=SMOOTHING_HISTORY_LENGTH)

# State variables
class GestureState:
    def __init__(self):
        self.left_click_active = False
        self.right_click_active = False
        self.window_swap_active = False
        self.last_window_switch_time = 0
        self.window_list = []
        self.current_window_index = 0
        self.prev_index_pos = (0, 0)
        self.prev_time = time.time()
        self.activation_confidence = 0
        self.cursor_stabilized = False
        self.cursor_reference_pos = None

state = GestureState()

def calculate_distance(pos1, pos2):
    """Calculate distance between two positions (either tuples or landmark objects)"""
    if hasattr(pos1, 'x'):  # If it's a landmark object
        x1, y1 = pos1.x, pos1.y
    else:  # If it's a tuple
        x1, y1 = pos1
        
    if hasattr(pos2, 'x'):  # If it's a landmark object
        x2, y2 = pos2.x, pos2.y
    else:  # If it's a tuple
        x2, y2 = pos2
        
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def get_smoothed_cursor_position(current_pos):
    """Apply smoothing to cursor position using moving average"""
    # Convert landmark to tuple if needed
    if hasattr(current_pos, 'x'):
        current_pos = (current_pos.x, current_pos.y)
    
    cursor_history.append(current_pos)
    
    if len(cursor_history) < SMOOTHING_HISTORY_LENGTH // 2:
        return current_pos
    
    # Calculate weighted average (more weight to recent positions)
    weights = np.linspace(0.1, 1.0, len(cursor_history))
    weights /= weights.sum()
    
    x = sum(p[0] * w for p, w in zip(cursor_history, weights))
    y = sum(p[1] * w for p, w in zip(cursor_history, weights))
    
    return (x, y)

def refresh_window_list():
    """Get all visible windows and filter out empty titles"""
    try:
        windows = getWindowsWithTitle("")
        state.window_list = [win for win in windows if win.title.strip()]
        print(f"Detected {len(state.window_list)} windows")
    except Exception as e:
        print(f"Error getting windows: {e}")
        state.window_list = []

def switch_window(direction):
    """Switch window with debounce and proper activation"""
    current_time = time.time()
    if current_time - state.last_window_switch_time < DEBOUNCE_TIME:
        return
    
    if not state.window_list:
        refresh_window_list()
        if not state.window_list:
            # Fallback to Alt+Tab
            pyautogui.keyDown('alt')
            if direction == "right":
                pyautogui.press('tab')
            else:
                pyautogui.keyDown('shift')
                pyautogui.press('tab')
                pyautogui.keyUp('shift')
            time.sleep(0.1)
            pyautogui.keyUp('alt')
            state.last_window_switch_time = current_time
            return
    
    try:
        if direction == "right":
            state.current_window_index = (state.current_window_index + 1) % len(state.window_list)
        else:
            state.current_window_index = (state.current_window_index - 1) % len(state.window_list)
        
        target_window = state.window_list[state.current_window_index]
        target_window.activate()
        state.last_window_switch_time = current_time
        print(f"Switched to: {target_window.title}")
        
    except Exception as e:
        print(f"Window switch error: {e}")
        refresh_window_list()

def draw_window_info(frame, index_pos):
    """Draw current window information on frame"""
    if not state.window_list:
        return
    
    # Convert position to tuple if it's a landmark
    if hasattr(index_pos, 'x'):
        index_pos = (index_pos.x, index_pos.y)
    
    panel_width = min(400, frame.shape[1] - 20)
    panel_height = 80
    panel_x = int(index_pos[0] * frame.shape[1] - panel_width//2)
    panel_y = int(index_pos[1] * frame.shape[0] - panel_height - 50)
    
    panel_x = max(10, min(panel_x, frame.shape[1] - panel_width - 10))
    panel_y = max(10, min(panel_y, frame.shape[0] - panel_height - 10))
    
    cv2.rectangle(frame, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (50, 50, 50), -1)
    cv2.rectangle(frame, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (255, 255, 255), 2)
    
    current_title = state.window_list[state.current_window_index].title[:35] + "..." if len(state.window_list[state.current_window_index].title) > 35 else state.window_list[state.current_window_index].title
    cv2.putText(frame, f"Window: {current_title}", 
               (panel_x + 10, panel_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Swipe Left/Right to Change ({state.current_window_index + 1}/{len(state.window_list)})", 
               (panel_x + 10, panel_y + 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Initial window list refresh
refresh_window_list()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = hands.process(frame_rgb)
        
        current_time = time.time()
        delta_time = current_time - state.prev_time
        state.prev_time = current_time
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get landmark positions
                landmarks = hand_landmarks.landmark
                
                # Get finger tips
                index_tip = landmarks[8]
                thumb_tip = landmarks[4]
                middle_tip = landmarks[12]
                pinky_tip = landmarks[20]
                wrist = landmarks[0]
                
                # Current positions
                current_index_pos = (index_tip.x, index_tip.y)
                
                # Calculate distances
                index_thumb_dist = calculate_distance(index_tip, thumb_tip)
                index_middle_dist = calculate_distance(index_tip, middle_tip)
                thumb_pinky_dist = calculate_distance(thumb_tip, pinky_tip)
                index_wrist_dist = calculate_distance(index_tip, wrist)
                
                # Cursor stabilization
                if not state.cursor_stabilized:
                    if state.cursor_reference_pos is None:
                        state.cursor_reference_pos = current_index_pos
                    else:
                        # Check if hand has stayed relatively still
                        if calculate_distance(current_index_pos, state.cursor_reference_pos) < 0.05:
                            state.cursor_stabilized = True
                            print("Cursor stabilized")
                        else:
                            state.cursor_reference_pos = current_index_pos
                
                # Apply smoothing
                smoothed_pos = get_smoothed_cursor_position(index_tip)
                
                # Move cursor with smoothing
                cursor_x = int(smoothed_pos[0] * screen_width)
                cursor_y = int(smoothed_pos[1] * screen_height)
                pyautogui.moveTo(cursor_x, cursor_y, duration=0.1)
                
                # Left click
                if index_thumb_dist < CLICK_THRESHOLD and not state.left_click_active:
                    pyautogui.click(button='left')
                    state.left_click_active = True
                    cv2.putText(frame, "Left Click", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif index_thumb_dist >= CLICK_THRESHOLD:
                    state.left_click_active = False
                
                # Right click
                if index_middle_dist < CLICK_THRESHOLD and not state.right_click_active:
                    pyautogui.click(button='right')
                    state.right_click_active = True
                    cv2.putText(frame, "Right Click", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif index_middle_dist >= CLICK_THRESHOLD:
                    state.right_click_active = False
                
                # Window swap mode
                if index_wrist_dist > WINDOW_SWAP_ACTIVATION_DISTANCE:
                    if not state.window_swap_active:
                        state.window_swap_active = True
                        print("Window swap mode activated")
                        refresh_window_list()
                    
                    draw_window_info(frame, index_tip)
                    
                    if state.prev_index_pos != (0, 0):
                        dx = current_index_pos[0] - state.prev_index_pos[0]
                        if abs(dx) > SWIPE_MIN_DISTANCE:
                            if dx > 0:
                                switch_window("right")
                            else:
                                switch_window("left")
                else:
                    if state.window_swap_active:
                        print("Window swap mode deactivated")
                    state.window_swap_active = False
                
                # Regular swipe gestures
                if not state.window_swap_active and thumb_pinky_dist < SWIPE_THRESHOLD:
                    if state.prev_index_pos != (0, 0):
                        dx = current_index_pos[0] - state.prev_index_pos[0]
                        dy = current_index_pos[1] - state.prev_index_pos[1]
                        distance = math.sqrt(dx**2 + dy**2)
                        
                        if distance > SWIPE_MIN_DISTANCE:
                            if abs(dx) > abs(dy):  # Horizontal swipe
                                if dx > 0:
                                    switch_window("right")
                                else:
                                    switch_window("left")
                            else:  # Vertical swipe
                                if dy > 0:
                                    pyautogui.hotkey('win', 'd')  # Show desktop
                                else:
                                    pyautogui.hotkey('win', 'tab')  # Task view
                
                state.prev_index_pos = current_index_pos
        
        # Display instructions
        cv2.putText(frame, "Left Click: Index + Thumb", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Right Click: Index + Middle", (50, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
         # cv2.putText(frame, "Quick Swipe: Thumb + Pinky", (50, 210), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Window Swap: Raise Hand High", (50, 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display stabilization status
        stability_status = "Stable" if state.cursor_stabilized else "Calibrating..."
        cv2.putText(frame, f"Cursor: {stability_status}", 
                   (frame.shape[1] - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if state.cursor_stabilized else (0, 0, 255), 1)
        
        # Display the frame
        cv2.imshow('Stable Hand Gesture Control', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


    
