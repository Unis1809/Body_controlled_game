import cv2
import mediapipe as mp
import pygame
import sys
import random
import math

# Initialize MediaPipe Pose and Face Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Body Controlled Game")

# Load character
character = pygame.Rect(375, 500, 50, 50)  # x, y, width, height

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GOLD = (255, 215, 0)

# Fonts
font_large = pygame.font.Font(None, 74)
font_small = pygame.font.Font(None, 36)

# Load sound
clap_sound = pygame.mixer.Sound("C:\\Users\\king\\Downloads\\jersey-type-snap-drum_A#_minor.wav")
background_music = "C:\\Users\\king\\Downloads\\01. Ground Theme.mp3"
pygame.mixer.music.load(background_music)
pygame.mixer.music.play(-1)  # Play background music in a loop

# Set up the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()


# Function to draw character
def draw_character(screen, character):
    pygame.draw.rect(screen, BLUE, character)


# Function to draw enemies
def draw_enemies(screen, enemies):
    for enemy in enemies:
        pygame.draw.rect(screen, RED, enemy)


# Function to draw power-ups
def draw_powerups(screen, powerups):
    for powerup in powerups:
        pygame.draw.rect(screen, GOLD, powerup)


# Function to check for collision
def check_collision(rect1, rect2_list):
    for rect2 in rect2_list:
        if rect1.colliderect(rect2):
            return rect2
    return None


# Generate a new enemy
def generate_enemy():
    x_pos = random.randint(0, 750)
    return pygame.Rect(x_pos, 0, 50, 50)  # x, y, width, height


# Generate a new power-up
def generate_powerup():
    x_pos = random.randint(0, 750)
    return pygame.Rect(x_pos, 0, 50, 50)  # x, y, width, height


# Function to display game over message
def display_game_over(screen, score):
    game_over_text = font_large.render('Game Over', True, BLACK)
    score_text = font_small.render(f'Score: {score}', True, BLACK)
    screen.blit(game_over_text, (250, 250))
    screen.blit(score_text, (350, 320))


# Function to display score
def display_score(screen, score):
    score_text = font_small.render(f'Score: {score}', True, BLACK)
    screen.blit(score_text, (10, 10))


# Function to detect hand clap
def detect_hand_clap(results):
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        hand1 = results.multi_hand_landmarks[0].landmark
        hand2 = results.multi_hand_landmarks[1].landmark

        # Get coordinates of the index finger tips
        hand1_index_tip = hand1[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        hand2_index_tip = hand2[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Calculate distance between index finger tips
        distance = math.sqrt((hand1_index_tip.x - hand2_index_tip.x) ** 2 +
                             (hand1_index_tip.y - hand2_index_tip.y) ** 2 +
                             (hand1_index_tip.z - hand2_index_tip.z) ** 2)

        # Threshold distance for detecting clap (tune as necessary)
        if distance < 0.05:
            return True
    return False


# Function to display direction or clapping text
def display_direction_text(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (50, 50), font, 2, (0, 255, 0), 3, cv2.LINE_AA)


# Main game loop
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
        mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
        mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    enemies = []
    powerups = []
    enemy_timer = 0
    powerup_timer = 0
    game_over = False
    score = 0
    enemy_speed = 5
    powerup_speed = 3
    direction_text = ""
    invincible = False
    invincible_timer = 0

    while True:
        # Capture frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Flip the frame horizontally for a later selfie-view display, and convert the BGR image to RGB.
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect the pose, hands, and face
        pose_results = pose.process(frame_rgb)
        hands_results = hands.process(frame_rgb)
        face_results = face_detection.process(frame_rgb)

        # Extract landmarks for pose
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark

            # Use nose position for horizontal movement
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            nose_x = int(nose.x * 800)  # Assuming the screen width is 800 pixels

            # Map nose x position to character x position
            if not game_over:
                character.x = nose_x - character.width // 2

                # Update direction text based on nose position
                if nose_x < 300:
                    direction_text = "Left"
                else:
                    direction_text = "Right"

        # Detect hand clap
        if detect_hand_clap(hands_results):
            clap_sound.play()
            direction_text = "Clapping"

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()

        if not game_over:
            # Update enemy and power-up timers and generate new enemies and power-ups
            enemy_timer += 1
            powerup_timer += 1
            if enemy_timer > 50:
                enemies.append(generate_enemy())
                enemy_timer = 0
            if powerup_timer > 200:
                powerups.append(generate_powerup())
                powerup_timer = 0

            # Move enemies and power-ups downwards
            for enemy in enemies:
                enemy.y += enemy_speed
            for powerup in powerups:
                powerup.y += powerup_speed

            # Check for collision with enemies
            collided_enemy = check_collision(character, enemies)
            if collided_enemy:
                if not invincible:
                    game_over = True
                else:
                    enemies.remove(collided_enemy)

            # Check for collision with power-ups
            collided_powerup = check_collision(character, powerups)
            if collided_powerup:
                powerups.remove(collided_powerup)
                invincible = True
                invincible_timer = 30  # Invincible for a period

            # Decrease invincible timer
            if invincible:
                invincible_timer -= 1
                if invincible_timer <= 0:
                    invincible = False

            # Increase score and difficulty
            score += 1
            if score % 1000 == 0:
                enemy_speed += 10

            # Clear screen
            screen.fill(WHITE)

            # Draw the character, enemies, and power-ups
            draw_character(screen, character)
            draw_enemies(screen, enemies)
            draw_powerups(screen, powerups)
            display_score(screen, score)

            # Remove off-screen enemies and power-ups
            enemies = [enemy for enemy in enemies if enemy.y < 600]
            powerups = [powerup for powerup in powerups if powerup.y < 600]
        else:
            # Display Game Over message
            display_game_over(screen, score)

        # Display direction text on the frame
        if direction_text:
            display_direction_text(frame, direction_text)

        # Draw face detection results
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)

        # Update display
        pygame.display.flip()

        # Show the camera feed
        cv2.imshow('MediaPipe Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.quit()
