import pygame
import numpy as np
import random
import time
import math
from scipy.stats import vonmises
import csv

# Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
TRIALS_PER_CONDITION = 30
CIRCLE_RADIUS = 200
CIRCLE_CENTER = (WIDTH // 2, HEIGHT // 2)
KAPPA = 7.4  # Concentration parameter for von Mises distribution

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)   # to enter into fullscreen mode
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Visual Estimate Task')
clock = pygame.time.Clock()
pygame.init()
pygame.font.init() 

# Function to display text on screen
def display_text(text, font_size, color, x, y):
    font = pygame.font.Font(None, font_size)
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))

# Function to generate a random target angle on the circle
def generate_target():
    angle = random.uniform(0, 2 * math.pi)
    x = CIRCLE_CENTER[0] + CIRCLE_RADIUS * math.cos(angle)
    y = CIRCLE_CENTER[1] + CIRCLE_RADIUS * math.sin(angle)
    return (x, y), angle

def calculate_influence(participant_initial_angle, participant_final_angle, partner_angle):
    # Calculate the absolute angular displacement toward the partner's estimate
    displacement = abs(participant_final_angle - participant_initial_angle)

    # Calculate initial angular distance from partner's estimate
    initial_distance = abs(participant_initial_angle - partner_angle)

    # Calculate influence ratio (avoid division by zero)
    influence_ratio = displacement / initial_distance if initial_distance != 0 else 0

    return influence_ratio

# Function to save trial data to a CSV file
def save_trial_data(condition, trial_num, influence_ratio, participant_initial_angle, participant_final_angle, partner_angle):
    with open('experiment.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([condition, trial_num, influence_ratio, participant_initial_angle, participant_final_angle, partner_angle])


# Function to scatter random yellow dots (distractors) at 60 Hz
def scatter_dots(n):
    for _ in range(n):
        angle = random.uniform(0, 2 * math.pi)
        x = CIRCLE_CENTER[0] + CIRCLE_RADIUS * math.cos(angle)
        y = CIRCLE_CENTER[1] + CIRCLE_RADIUS * math.sin(angle)
        pygame.draw.circle(screen, YELLOW, (int(x), int(y)), 5)
        pygame.display.flip()
        pygame.time.delay(int(1000 / 60))  # 60 Hz, 16.67 ms per frame

# Function to get the closest point on the circumference based on a click
def closest_point_on_circle(click_pos):
    dx, dy = click_pos[0] - CIRCLE_CENTER[0], click_pos[1] - CIRCLE_CENTER[1]
    angle = math.atan2(dy, dx)
    x = CIRCLE_CENTER[0] + CIRCLE_RADIUS * math.cos(angle)
    y = CIRCLE_CENTER[1] + CIRCLE_RADIUS * math.sin(angle)
    return (int(x), int(y)), angle

def minor_arc_angles(angle1, angle2):
    # Normalize angles to 0 - 2π
    angle1 = angle1 % (2 * math.pi)
    angle2 = angle2 % (2 * math.pi)

    # Determine the shortest arc
    if (angle2 - angle1) % (2 * math.pi) < math.pi:
        return angle1, angle2
    else:
        return angle2, angle1

# Function to draw the minor arc for revision
def draw_allowed_arc(min_angle, max_angle):
    for i in range(int(min_angle * 100), int(max_angle * 100)):
        angle = i / 100.0
        x = CIRCLE_CENTER[0] + CIRCLE_RADIUS * math.cos(angle)
        y = CIRCLE_CENTER[1] + CIRCLE_RADIUS * math.sin(angle)
        pygame.draw.circle(screen, GREEN, (int(x), int(y)), 3)

# Update within_allowed_arc to use minor arc
def within_allowed_arc(angle, min_angle, max_angle):
    angle = angle % (2 * math.pi)
    min_angle, max_angle = minor_arc_angles(min_angle, max_angle)
    
    # Check if angle is within the minor arc
    if min_angle < max_angle:
        return min_angle <= angle <= max_angle
    else:
        return angle >= min_angle or angle <= max_angle

# Function for the participant to estimate the target position on the circle
def participant_estimate(allowed_arc=None):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, None
            if event.type == pygame.MOUSEBUTTONDOWN:
                click_pos = pygame.mouse.get_pos()
                estimate_pos, angle = closest_point_on_circle(click_pos)
                # Check if within the allowed arc if specified
                if allowed_arc and not within_allowed_arc(angle, *allowed_arc):
                    print("Clicked outside the allowed arc.")
                    continue
                return estimate_pos, angle
        clock.tick(60)

def generate_first_choice(participant_angle, confidence, high_confidence_threshold=5):
    if confidence >= high_confidence_threshold:
        # Draw from a uniform distribution centered around participant's estimate ±20°
        angle_offset = random.uniform(-math.radians(20), math.radians(20))
        partner_angle = participant_angle + angle_offset
    else:
        # Draw from a von Mises distribution centered on the target
        partner_angle = vonmises.rvs(KAPPA, loc=participant_angle)
    
    x = CIRCLE_CENTER[0] + CIRCLE_RADIUS * math.cos(partner_angle)
    y = CIRCLE_CENTER[1] + CIRCLE_RADIUS * math.sin(partner_angle)
    return (int(x), int(y)), partner_angle

# Function to generate the partner's second choice based on susceptibility
def generate_second_choice(participant_angle, partner_angle, condition, transition_lambda=0):
    def draw_influence(probabilities, ranges):
        choice = random.choices(ranges, probabilities)[0]
        return random.uniform(choice[0], choice[1])

    if condition == 'susceptible':
        influence = draw_influence([0.5, 0.2, 0.3], [(0.7, 1), (0.3, 0.7), (0, 0.3)])
    elif condition == 'insusceptible':
        influence = draw_influence([0.65, 0.2, 0.15], [(0, 0.2), (0.3, 0.7), (0.7, 0.9)])
    elif condition == 'baseline':
        # Interpolate between susceptible and insusceptible conditions
        susceptible_influence = draw_influence([0.5, 0.2, 0.3], [(0.7, 1), (0.3, 0.7), (0, 0.3)])
        insusceptible_influence = draw_influence([0.65, 0.2, 0.15], [(0, 0.2), (0.3, 0.7), (0.7, 0.9)])
        influence = (1 - transition_lambda) * susceptible_influence + transition_lambda * insusceptible_influence
    
    # Calculate the new partner angle
    new_partner_angle = partner_angle + influence * (participant_angle - partner_angle)
    x = CIRCLE_CENTER[0] + CIRCLE_RADIUS * math.cos(new_partner_angle)
    y = CIRCLE_CENTER[1] + CIRCLE_RADIUS * math.sin(new_partner_angle)
    return (int(x), int(y)), new_partner_angle


# Function for the partner's estimate based on condition
def partner_estimate(participant_angle, condition, confidence=None, transition_lambda=0):
    partner_first_choice, partner_first_angle = generate_first_choice(participant_angle, confidence or 1)
    
    # Generate the second choice only if necessary
    partner_second_choice, partner_second_angle = generate_second_choice(
        participant_angle, partner_first_angle, condition, transition_lambda
    )
    return partner_first_choice, partner_first_angle, partner_second_choice, partner_second_angle

def welcome_screen():
    screen.fill(BLACK)
    display_text('Welcome to the Visual Estimate Experiment', 36, WHITE, 100, HEIGHT // 3)
    display_text('In this task, you will estimate target locations on the circle.', 28, WHITE, 100, HEIGHT // 3 + 50)
    display_text('Your partner will also provide estimates.', 28, WHITE, 100, HEIGHT // 3 + 90)
    display_text('Press Enter to begin.', 36, GREEN, 100, HEIGHT // 3 + 150)
    pygame.display.flip()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                waiting = False
                return True
    return False

def run_experiment():
    if not welcome_screen():
        return
    
    conditions = ['susceptible', 'baseline', 'insusceptible']
    alternate_turn = True  # Track turns in susceptible and insusceptible conditions
    
        

    for condition in conditions:
        print(f"Running trials for condition: {condition}")

        for trial in range(TRIALS_PER_CONDITION):
            print(f"\nTrial {trial + 1}/{TRIALS_PER_CONDITION} for {condition}")

            # Display a message indicating the next trial is starting
            screen.fill(BLACK)
            display_text('Starting the next trial...', 36, WHITE, 100, HEIGHT // 2)
            pygame.display.flip()
            time.sleep(2)
        
        

            # 1. Display the true target for 25 ms
            true_position, true_angle = generate_target()
            screen.fill(BLACK)
            pygame.draw.circle(screen, WHITE, CIRCLE_CENTER, CIRCLE_RADIUS, 2)
            pygame.draw.circle(screen, YELLOW, true_position, 10)
            pygame.display.flip()
            pygame.time.delay(100)

            # 2. Scatter 90 random distractor dots at 60 Hz
            screen.fill(BLACK)
            pygame.draw.circle(screen, WHITE, CIRCLE_CENTER, CIRCLE_RADIUS, 2)
            scatter_dots(90)

            # 3. Clear the screen
            screen.fill(BLACK)
            pygame.draw.circle(screen, WHITE, CIRCLE_CENTER, CIRCLE_RADIUS, 2)
            pygame.display.flip()
            time.sleep(1)

            # 4. Participant makes an initial estimate
            display_text('Click on the circle to estimate the target location.', 36, WHITE, 100, 50)
            pygame.display.flip()
            participant_est, participant_angle = participant_estimate()
            print(f"Participant estimate: {participant_est}, Angle: {participant_angle}")

            # 5. Prompt for confidence selection from 1 to 9
            screen.fill(BLACK)
            pygame.draw.circle(screen, WHITE, CIRCLE_CENTER, CIRCLE_RADIUS, 2)
            display_text('Select your confidence on the target from 1-6', 36, WHITE, 100, 50)
            pygame.display.flip()

            confidence = None
            while confidence is None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return None
                    if event.type == pygame.KEYDOWN:
                        if pygame.K_1 <= event.key <= pygame.K_6:
                            confidence = event.key - pygame.K_0
                            print(f"Confidence level: {confidence}")
                            break

            # 6. Generate partner's initial estimate in all conditions
            partner_first_choice, partner_first_angle, partner_second_choice, partner_second_angle = partner_estimate(participant_angle, condition)
            if partner_first_choice:
                print(f"Partner estimate: {partner_first_choice}, Angle: {partner_first_angle}")

            # Display both estimates if available
            screen.fill(BLACK)
            pygame.draw.circle(screen, WHITE, CIRCLE_CENTER, CIRCLE_RADIUS, 2)
            pygame.draw.circle(screen, BLUE, participant_est, 10)  # Participant's estimate
            time.sleep(3)
            pygame.draw.circle(screen, RED, partner_first_choice, 10)  # Partner's estimate
            display_text("Blue: Your Estimate, Red: Partner Estimate", 30, WHITE, 50, HEIGHT - 50)
            pygame.display.flip()
            time.sleep(2)

            final_estimate = participant_est

            # Revision logic based on condition
            if condition == 'baseline':
                display_text('Do you want to revise your estimate? (Y/N)', 36, WHITE, 100, 50)
                pygame.display.flip()
                revising = None
                while revising is None:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return None
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_y:
                                revising = True
                            elif event.key == pygame.K_n:
                                revising = False

                if revising:
                    screen.fill(BLACK)
                    pygame.draw.circle(screen, WHITE, CIRCLE_CENTER, CIRCLE_RADIUS, 2)
                    min_angle, max_angle = minor_arc_angles(participant_angle, partner_first_angle)
                    draw_allowed_arc(min_angle, max_angle)
                    display_text('Click within the allowed arc to revise your estimate.', 36, WHITE, 100, 50)
                    pygame.display.flip()
                    allowed_arc = (min_angle, max_angle)
                    revised_est, revised_angle = participant_estimate(allowed_arc)
                    final_estimate = revised_est

            else:
                if alternate_turn:
                    display_text('Do you want to revise your estimate? (Y/N)', 36, WHITE, 100, 50)
                    pygame.display.flip()
                    revising = None
                    while revising is None:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                return None
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_y:
                                    revising = True
                                elif event.key == pygame.K_n:
                                    revising = False

                    if revising:
                        screen.fill(BLACK)
                        pygame.draw.circle(screen, WHITE, CIRCLE_CENTER, CIRCLE_RADIUS, 2)
                        min_angle, max_angle = minor_arc_angles(participant_angle, partner_first_angle)
                        draw_allowed_arc(min_angle, max_angle)
                        display_text('Click within the allowed arc to revise your estimate.', 36, WHITE, 100, 50)
                        pygame.display.flip()
                        allowed_arc = (min_angle, max_angle)
                        revised_est, revised_angle = participant_estimate(allowed_arc)
                        final_estimate = revised_est
                else:
                    time.sleep(3)
                    partner_revised_est = partner_second_choice
                    screen.fill(BLACK)
                    pygame.draw.circle(screen, WHITE, CIRCLE_CENTER, CIRCLE_RADIUS, 2)
                    pygame.draw.circle(screen, RED, partner_revised_est, 10)
                    display_text("Red: Partner's Revised Estimate", 30, WHITE, 50, HEIGHT - 50)
                    pygame.display.flip()
                    time.sleep(2)

            alternate_turn = not alternate_turn

            screen.fill(BLACK)
            pygame.draw.circle(screen, WHITE, CIRCLE_CENTER, CIRCLE_RADIUS, 2)
            pygame.draw.circle(screen, BLUE, final_estimate, 10)
            display_text("Blue: Final Estimate", 30, WHITE, 50, HEIGHT - 50)
            pygame.display.flip()
            time.sleep(2)

        # Calculate influence and save trial data
            influence_ratio = calculate_influence(participant_angle, revised_angle if revising else participant_angle, partner_first_angle)
            save_trial_data(condition, trial + 1, influence_ratio, participant_angle, revised_angle if revising else participant_angle, partner_first_angle)
            

    pygame.quit()
# Run the experiment
run_experiment()

