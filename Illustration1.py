import pygame
import sys
import random
import math
from pygame.locals import *

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1000, 700
NUM_CIRCLES = 8
BALLS_PER_CIRCLE = 24
BALL_RADIUS = 8
CIRCLE_SPACING = 40
CENTER_X = WIDTH // 2
CENTER_Y = HEIGHT // 2
SLIDER_HEIGHT = 30
SLIDER_WIDTH = 400
SLIDER_X = (WIDTH - SLIDER_WIDTH) // 2
SLIDER_Y = HEIGHT - 100

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (230, 230, 230)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
COLORS = [RED, GREEN, BLUE, YELLOW, PURPLE, CYAN, ORANGE]
COLOR_NAMES = ["Red", "Green", "Blue", "Yellow", "Purple", "Cyan", "Orange"]

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Color Ball Physics Simulation")
font = pygame.font.SysFont('Arial', 16)
title_font = pygame.font.SysFont('Arial', 20, bold=True)

class Ball:
    def __init__(self, x, y, color, circle_idx, ball_idx):
        self.base_x = x
        self.base_y = y
        self.base_z = 0
        self.color = color
        self.circle_idx = circle_idx
        self.ball_idx = ball_idx
        self.height = 0
        self.target_height = 0
        self.random_factor = random.uniform(0.95, 1.05)  # 5% random variation

        # Store initial values for reset
        self.initial_height = 0
        self.initial_target_height = 0

    def update(self):
        # Smoothly transition to target height
        self.height += (self.target_height - self.height) * 0.1

    def reset(self):
        self.height = self.initial_height
        self.target_height = self.initial_target_height

    def draw(self, surface, angle_x, angle_y, angle_z, zoom, draw_lines=False):
        # Get the current position
        x, y, z = self.base_x, self.base_y, self.base_z + self.height

        # Apply rotation around X axis
        y_rot = y * math.cos(angle_x) - z * math.sin(angle_x)
        z_rot = y * math.sin(angle_x) + z * math.cos(angle_x)
        y = y_rot
        z = z_rot

        # Apply rotation around Y axis
        x_rot = x * math.cos(angle_y) + z * math.sin(angle_y)
        z_rot = -x * math.sin(angle_y) + z * math.cos(angle_y)
        x = x_rot
        z = z_rot

        # Apply rotation around Z axis
        x_rot = x * math.cos(angle_z) - y * math.sin(angle_z)
        y_rot = x * math.sin(angle_z) + y * math.cos(angle_z)
        x = x_rot
        y = y_rot

        # Apply zoom and center
        scale = zoom / (zoom + z) if (zoom + z) != 0 else 1
        x_proj = x * scale + CENTER_X
        y_proj = y * scale + CENTER_Y
        radius_proj = max(2, BALL_RADIUS * scale)

        # Draw connection line to vertex if enabled
        if draw_lines and self.height > 5:
            # Calculate vertex position (center point at z=0)
            vx, vy, vz = 0, 0, 0

            # Apply rotation to vertex
            vy_rot = vy * math.cos(angle_x) - vz * math.sin(angle_x)
            vz_rot = vy * math.sin(angle_x) + vz * math.cos(angle_x)
            vy = vy_rot
            vz = vz_rot

            vx_rot = vx * math.cos(angle_y) + vz * math.sin(angle_y)
            vz_rot = -vx * math.sin(angle_y) + vz * math.cos(angle_y)
            vx = vx_rot
            vz = vz_rot

            vx_rot = vx * math.cos(angle_z) - vy * math.sin(angle_z)
            vy_rot = vx * math.sin(angle_z) + vy * math.cos(angle_z)
            vx = vx_rot
            vy = vy_rot

            # Project vertex
            v_scale = zoom / (zoom + vz) if (zoom + vz) != 0 else 1
            vx_proj = vx * v_scale + CENTER_X
            vy_proj = vy * v_scale + CENTER_Y

            # Draw line from vertex to ball
            line_color = (self.color[0]//2, self.color[1]//2, self.color[2]//2)
            pygame.draw.line(surface, line_color, (vx_proj, vy_proj), (x_proj, y_proj), 1)

        # Draw the ball with shading based on z-depth
        shade = max(0.4, min(1.0, 1 - z/1000))
        shaded_color = (
            int(self.color[0] * shade),
            int(self.color[1] * shade),
            int(self.color[2] * shade)
        )

        pygame.draw.circle(surface, shaded_color, (int(x_proj), int(y_proj)), int(radius_proj))

        # Draw a highlight
        if radius_proj > 5:
            highlight_pos = (int(x_proj - radius_proj * 0.3), int(y_proj - radius_proj * 0.3))
            highlight_radius = int(radius_proj * 0.3)
            pygame.draw.circle(surface, WHITE, highlight_pos, highlight_radius, 1)

class BallSystem:
    def __init__(self):
        self.balls = []
        self.selected_color = None
        self.base_height = 0
        self.draw_lines = True  # Toggle for connection lines

        # Create concentric circles of balls
        for circle_idx in range(NUM_CIRCLES):
            circle_radius = (circle_idx + 1) * CIRCLE_SPACING
            circle_balls = []

            for ball_idx in range(BALLS_PER_CIRCLE):
                angle = 2 * math.pi * ball_idx / BALLS_PER_CIRCLE
                x = circle_radius * math.cos(angle)
                y = circle_radius * math.sin(angle)
                color = random.choice(COLORS)

                ball = Ball(x, y, color, circle_idx, ball_idx)
                circle_balls.append(ball)

            self.balls.append(circle_balls)

    def update_heights(self, height_value):
        self.base_height = height_value
        for circle in self.balls:
            for ball in circle:
                if self.selected_color is None or ball.color == self.selected_color:
                    # Apply base height with random variation
                    ball.target_height = height_value * ball.random_factor * 20

    def update(self):
        for circle in self.balls:
            for ball in circle:
                ball.update()

    def reset(self):
        for circle in self.balls:
            for ball in circle:
                ball.reset()
        self.base_height = 0
        self.selected_color = None

    def toggle_lines(self):
        self.draw_lines = not self.draw_lines

    def draw(self, surface, angle_x, angle_y, angle_z, zoom):
        # Draw balls with connection lines if enabled
        for circle in self.balls:
            for ball in circle:
                ball.draw(surface, angle_x, angle_y, angle_z, zoom, self.draw_lines)

class Slider:
    def __init__(self, x, y, width, height, min_val=0, max_val=1, initial=0.0, label="", is_angle=False):
        self.rect = pygame.Rect(x, y, width, height)
        self.knob = pygame.Rect(x, y - 5, 20, height + 10)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.dragging = False
        self.label = label
        self.is_angle = is_angle
        self.update_knob()

    def update_knob(self):
        self.knob.centerx = self.rect.x + ((self.value - self.min_val) / (self.max_val - self.min_val)) * self.rect.width

    def set_value(self, value):
        self.value = max(self.min_val, min(self.max_val, value))
        self.update_knob()

    def reset(self):
        self.set_value(0)

    def handle_event(self, event):
        if event.type == MOUSEBUTTONDOWN:
            if self.knob.collidepoint(event.pos) or self.rect.collidepoint(event.pos):
                self.dragging = True
                if self.rect.collidepoint(event.pos):
                    self.set_value(self.min_val + (event.pos[0] - self.rect.x) / self.rect.width * (self.max_val - self.min_val))
                    return True
        elif event.type == MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == MOUSEMOTION and self.dragging:
            self.set_value(self.min_val + (event.pos[0] - self.rect.x) / self.rect.width * (self.max_val - self.min_val))
            return True
        return False

    def draw(self, surface):
        pygame.draw.rect(surface, LIGHT_GRAY, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 1)
        pygame.draw.rect(surface, BLACK, self.knob)

        # Draw label with value
        if self.is_angle:
            value_text = f"{math.degrees(self.value):.1f}°"
        else:
            value_text = f"{self.value:.2f}"

        text = font.render(f"{self.label}: {value_text}", True, BLACK)
        surface.blit(text, (self.rect.x, self.rect.y - 20))

class ColorSelector:
    def __init__(self, x, y, size=30):
        self.buttons = []
        self.selected_color = None
        for i, color in enumerate(COLORS):
            self.buttons.append({
                'rect': pygame.Rect(x + i * (size + 10), y, size, size),
                'color': color,
                'name': COLOR_NAMES[i]
            })

    def handle_event(self, event):
        if event.type == MOUSEBUTTONDOWN:
            for button in self.buttons:
                if button['rect'].collidepoint(event.pos):
                    if self.selected_color == button['color']:
                        self.selected_color = None  # Deselect if clicked again
                    else:
                        self.selected_color = button['color']
                    return True
        return False

    def reset(self):
        self.selected_color = None

    def draw(self, surface):
        text = font.render("Select Color:", True, BLACK)
        surface.blit(text, (self.buttons[0]['rect'].x, self.buttons[0]['rect'].y - 20))

        for button in self.buttons:
            pygame.draw.rect(surface, button['color'], button['rect'])
            pygame.draw.rect(surface, BLACK, button['rect'], 1)

            # Highlight selected color
            if self.selected_color == button['color']:
                pygame.draw.rect(surface, WHITE, button['rect'], 3)

class RotationControl:
    def __init__(self, x, y, radius=50):
        self.rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
        self.radius = radius
        self.center = (x, y)
        self.dragging = False
        self.angle_x = 0
        self.angle_y = 0

    def handle_event(self, event):
        if event.type == MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self.update_angles(event.pos)
                return True
        elif event.type == MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == MOUSEMOTION and self.dragging:
            self.update_angles(event.pos)
            return True
        return False

    def update_angles(self, pos):
        dx = pos[0] - self.center[0]
        dy = pos[1] - self.center[1]
        self.angle_x = dy / self.radius * 0.5
        self.angle_y = dx / self.radius * 0.5

    def draw(self, surface):
        # Draw control circle
        pygame.draw.circle(surface, LIGHT_GRAY, self.center, self.radius)
        pygame.draw.circle(surface, BLACK, self.center, self.radius, 1)

        # Draw crosshairs
        pygame.draw.line(surface, BLACK, (self.center[0] - self.radius, self.center[1]),
                         (self.center[0] + self.radius, self.center[1]), 1)
        pygame.draw.line(surface, BLACK, (self.center[0], self.center[1] - self.radius),
                         (self.center[0], self.center[1] + self.radius), 1)

        # Draw current position indicator
        indicator_x = self.center[0] + self.angle_y * self.radius * 2
        indicator_y = self.center[1] + self.angle_x * self.radius * 2
        pygame.draw.circle(surface, RED, (int(indicator_x), int(indicator_y)), 5)

        # Draw label
        text = font.render("Rotation Control", True, BLACK)
        surface.blit(text, (self.center[0] - text.get_width() // 2, self.center[1] + self.radius + 5))

class Button:
    def __init__(self, x, y, width, height, text, color=LIGHT_GRAY, hover_color=GRAY, text_color=BLACK):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False

    def handle_event(self, event):
        if event.type == MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(self, surface):
        # Draw button
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)

        # Draw text
        text_surface = font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

def main():
    clock = pygame.time.Clock()
    ball_system = BallSystem()

    # Create sliders
    height_slider = Slider(SLIDER_X, SLIDER_Y, SLIDER_WIDTH, SLIDER_HEIGHT, 0, 10, 0, "Height")
    rotate_x_slider = Slider(50, 50, 150, 15, -math.pi, math.pi, 0, "Rotate X", True)
    rotate_y_slider = Slider(50, 100, 150, 15, -math.pi, math.pi, 0, "Rotate Y", True)
    rotate_z_slider = Slider(50, 150, 150, 15, -math.pi, math.pi, 0, "Rotate Z", True)
    zoom_slider = Slider(WIDTH - 200, 50, 150, 15, 200, 1000, 600, "Zoom")

    # Create rotation control
    rotation_control = RotationControl(WIDTH - 100, HEIGHT - 200)

    color_selector = ColorSelector(SLIDER_X, SLIDER_Y + 50)

    # Create buttons
    reset_button = Button(WIDTH - 100, 100, 80, 30, "Reset")
    lines_button = Button(WIDTH - 100, 140, 80, 30, "Lines: ON")
    auto_rotate_button = Button(WIDTH - 100, 180, 80, 30, "Auto: OFF", text_color=RED)

    # Auto-rotation variables
    auto_rotate = False
    rotation_speed = 0.01

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            if height_slider.handle_event(event):
                ball_system.selected_color = color_selector.selected_color
                ball_system.update_heights(height_slider.value)

            rotate_x_slider.handle_event(event)
            rotate_y_slider.handle_event(event)
            rotate_z_slider.handle_event(event)
            zoom_slider.handle_event(event)
            color_selector.handle_event(event)
            rotation_control.handle_event(event)

            # Handle reset button
            if reset_button.handle_event(event):
                ball_system.reset()
                height_slider.reset()
                color_selector.reset()

            # Handle lines toggle button
            if lines_button.handle_event(event):
                ball_system.toggle_lines()
                lines_button.text = "Lines: ON" if ball_system.draw_lines else "Lines: OFF"

            # Handle auto-rotation button
            if auto_rotate_button.handle_event(event):
                auto_rotate = not auto_rotate
                auto_rotate_button.text = "Auto: ON" if auto_rotate else "Auto: OFF"
                auto_rotate_button.text_color = GREEN if auto_rotate else RED

            # Toggle auto-rotation with spacebar
            if event.type == KEYDOWN and event.key == K_SPACE:
                auto_rotate = not auto_rotate
                auto_rotate_button.text = "Auto: ON" if auto_rotate else "Auto: OFF"
                auto_rotate_button.text_color = GREEN if auto_rotate else RED

        # Update ball positions
        ball_system.update()

        # Handle auto-rotation
        if auto_rotate:
            rotate_x_slider.set_value(rotate_x_slider.value + rotation_speed)
            rotate_y_slider.set_value(rotate_y_slider.value + rotation_speed * 1.3)
            rotate_z_slider.set_value(rotate_z_slider.value + rotation_speed * 0.7)

        # Apply rotation from control
        if rotation_control.dragging:
            rotate_x_slider.set_value(rotate_x_slider.value + rotation_control.angle_x)
            rotate_y_slider.set_value(rotate_y_slider.value + rotation_control.angle_y)
            rotation_control.angle_x = 0
            rotation_control.angle_y = 0

        # Clear the screen
        screen.fill(WHITE)

        # Draw the ball system with current rotation and zoom
        ball_system.draw(screen, rotate_x_slider.value, rotate_y_slider.value,
                        rotate_z_slider.value, zoom_slider.value)

        # Draw the sliders
        height_slider.draw(screen)
        rotate_x_slider.draw(screen)
        rotate_y_slider.draw(screen)
        rotate_z_slider.draw(screen)
        zoom_slider.draw(screen)

        # Draw rotation control
        rotation_control.draw(screen)

        # Draw color selector
        color_selector.draw(screen)

        # Draw buttons
        reset_button.draw(screen)
        lines_button.draw(screen)
        auto_rotate_button.draw(screen)

        # Draw title
        title = title_font.render("3D Color Ball Physics Simulation", True, BLACK)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 10))

        # Draw minimal instructions
        instructions = [
            "Height: Move balls up/down",
            "Rotate: Change view angle",
            "Zoom: Adjust perspective",
            "Space: Toggle auto-rotation"
        ]

        for i, instruction in enumerate(instructions):
            text = font.render(instruction, True, BLACK)
            screen.blit(text, (10, HEIGHT - 100 + i * 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
