import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.models import SimpleModel, AdvancedModel

#--------------------------------------------------------
#-                      MODEL CODE                      -
#--------------------------------------------------------

SIMPLE_MODEL_NAME = "Simple/Model_Epoch10_LR0_001.pth"
ADVANCED_MODEL_NAME = "Advanced/Model_Epoch5_LR0_001.pth"

simpleModel = SimpleModel()
advancedModel = AdvancedModel()
simpleModel.load_state_dict(torch.load(f"./models/{SIMPLE_MODEL_NAME}", weights_only=True))
advancedModel.load_state_dict(torch.load(f"./models/{ADVANCED_MODEL_NAME}", weights_only=True))
simpleModel.eval()
advancedModel.eval()

#--------------------------------------------------------
#-                  INTERACTIVE PLOT                    -
#--------------------------------------------------------

grid = torch.zeros((1, 28, 28), dtype=torch.float32)
brush_size = 2
drawing = False

fig, ax = plt.subplots(figsize=(7, 4))
plt.subplots_adjust(left=0.2, bottom=0.25)

try:
    manager = plt.get_current_fig_manager()
    manager.window.resizable(False, False)
except Exception as e:
    pass

im = ax.imshow(grid[0], cmap='gray', vmin=0, vmax=1)
ax.set_xticks([])
ax.set_yticks([])

simple_confidence_boxes = []
advanced_confidence_boxes = []
y_spacing = 0.045
label_y = 0.93
start_y = 0.88
simple_x = 0.05
advanced_x = 0.82

fig.text(simple_x, label_y, "Simple", fontsize=12, fontweight='bold', )
fig.text(advanced_x, label_y, "Advanced", fontsize=12, fontweight='bold')

for i in range(10):
    y = start_y - i * y_spacing
    text = fig.text(simple_x, y, f"{i}: 0.00", fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    simple_confidence_boxes.append(text)

for i in range(10):
    y = start_y - i * y_spacing
    text = fig.text(advanced_x, y, f"{i}: 0.00", fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    advanced_confidence_boxes.append(text)


def simple_update_confidence():
    simple_conf_scores = F.softmax(simpleModel(grid), dim=1).reshape(10, 1)
    max_index = torch.argmax(simple_conf_scores)
    for i in range(10):
        simple_confidence_boxes[i].set_text(f"{i}: {simple_conf_scores[i].item():.2f}")
        if i == max_index:
            simple_confidence_boxes[i].set_bbox(dict(facecolor='lightgreen', edgecolor='black'))
        else:
            simple_confidence_boxes[i].set_bbox(dict(facecolor='white', edgecolor='black'))

def adv_update_confidence():
    adv_conf_scores = F.softmax(advancedModel(grid), dim=1).reshape(10, 1)
    max_index = torch.argmax(adv_conf_scores)
    for i in range(10):
        advanced_confidence_boxes[i].set_text(f"{i}: {adv_conf_scores[i].item():.2f}")
        if i == max_index:
            advanced_confidence_boxes[i].set_bbox(dict(facecolor='lightgreen', edgecolor='black'))
        else:
            advanced_confidence_boxes[i].set_bbox(dict(facecolor='white', edgecolor='black'))

def get_brush_kernel(size=5):
    assert size % 2 == 1
    radius = size // 2
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    sigma = radius / 2
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.max()
    return kernel

def paint(x, y):
    i, j = int(y), int(x)
    kernel = get_brush_kernel(size=brush_size*2+1)

    radius = brush_size
    _, h, w = grid.shape

    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                grid[0, ni, nj] = min(grid[0, ni, nj] + kernel[di+radius, dj+radius], 1.0)

    simple_update_confidence()
    adv_update_confidence()

def on_press(event):
    global drawing
    if event.inaxes != ax:
        return
    drawing = True
    im.set_data(grid[0])
    fig.canvas.draw_idle()

def on_release(event):
    global drawing
    drawing = False

def on_motion(event):
    if drawing and event.inaxes == ax:
        paint(event.xdata, event.ydata)
        im.set_data(grid[0])
        fig.canvas.draw_idle()

def clear(event):
    global grid
    grid = torch.zeros((1, 28, 28), dtype=torch.float32)
    simple_update_confidence()
    adv_update_confidence()
    im.set_data(grid[0])
    fig.canvas.draw_idle()

clear_ax = plt.axes([0.4, 0.05, 0.2, 0.06])
clear_button = Button(clear_ax, 'Clear', color='lightgray', hovercolor='gray')
clear_button.on_clicked(clear)

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.manager.set_window_title('MNIST Digit Interactive Guesser')

plt.show()