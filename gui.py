
import pygame
import numpy as np
import subprocess
import math


WIDTH, HEIGHT = 280, 280
ROWS, COLS = 28, 28
CELL_SIZE = WIDTH // COLS

pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SPACE to guess, C to clear")

grid = np.zeros((ROWS, COLS), dtype=float)

def draw_grid():
    win.fill((0, 0, 0))
    for row in range(ROWS):
        for col in range(COLS):
            val = int(grid[row][col])
            if val > 0:
                pygame.draw.rect(win, (val, val, val), (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

def paint(row, col):
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            r = row + dr
            c = col + dc
            
            if 0 <= r < ROWS and 0 <= c < COLS:
                dist_sq = dr**2 + dc**2
                
                if dist_sq == 0:
                    intensity = 255.0 
                elif dist_sq == 1:
                    intensity = 60.0   
                else:
                    intensity = 0.0  
                
                if intensity > 0:
                    grid[r][c] = max(grid[r][c], intensity)


def interpolate_and_paint(r1, c1, r2, c2):
  
    distance = max(1, int(math.hypot(r2 - r1, c2 - c1)))
    for i in range(distance + 1):
        r = r1 + (r2 - r1) * (i / distance)
        c = c1 + (c2 - c1) * (i / distance)
        paint(int(round(r)), int(round(c)))

def predict():
    global grid
    
    nonzero_coords = np.argwhere(grid > 0)
    if len(nonzero_coords) == 0:
        return
        
    y_min, x_min = nonzero_coords.min(axis=0)
    y_max, x_max = nonzero_coords.max(axis=0)
    cropped = grid[y_min:y_max+1, x_min:x_max+1]
    
    h, w = cropped.shape
    scale_factor = 20.0 / max(h, w)
    new_w, new_h = max(1, int(w * scale_factor)), max(1, int(h * scale_factor))
    
    rgb_cropped = np.stack((cropped.T,) * 3, axis=-1)

    surf = pygame.surfarray.make_surface(rgb_cropped)
    scaled_surf = pygame.transform.smoothscale(surf, (new_w, new_h))
    
    resized = pygame.surfarray.pixels3d(scaled_surf)[..., 0].T
    
    y_indices, x_indices = np.indices(resized.shape)
    total_mass = np.sum(resized)
    
    if total_mass > 0:
        cy = np.sum(y_indices * resized) / total_mass
        cx = np.sum(x_indices * resized) / total_mass
    else:
        cy, cx = new_h / 2.0, new_w / 2.0
        
    final_grid = np.zeros((28, 28), dtype=float)
    
    start_y = int(round(13.5 - cy))
    start_x = int(round(13.5 - cx))
    
    for i in range(new_h):
        for j in range(new_w):
            y_pos = start_y + i
            x_pos = start_x + j
            if 0 <= y_pos < 28 and 0 <= x_pos < 28:
                final_grid[y_pos, x_pos] = resized[i, j]
                
    grid[:, :] = final_grid
    
    normalized_grid = final_grid / 255.0
    np.savetxt("input.txt", normalized_grid.flatten(), fmt='%.6f')
    
    try:
        result = subprocess.run(['.\\nn.exe', 'predict'], capture_output=True, text=True)
        prediction = result.stdout.strip()
        print(f"~{prediction}")
    except Exception as e:
        print(f"Couldn`t find nn.exe executable")

running = True
drawing = False
last_r, last_c = None, None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                drawing = True
                pos = pygame.mouse.get_pos()
                last_r, last_c = pos[1] // CELL_SIZE, pos[0] // CELL_SIZE
                paint(last_r, last_c)
                
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
                last_r, last_c = None, None
                
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                pos = pygame.mouse.get_pos()
                r, c = pos[1] // CELL_SIZE, pos[0] // CELL_SIZE
                if last_r is not None and last_c is not None:
                    interpolate_and_paint(last_r, last_c, r, c)
                last_r, last_c = r, c
                
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                predict()
            if event.key == pygame.K_c:
                grid.fill(0)

    draw_grid()
    pygame.display.update()

pygame.quit()