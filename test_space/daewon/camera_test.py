import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data as pd
from PIL import Image

def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
p.loadURDF('plane.urdf')
p.loadURDF('cube_small.urdf', basePosition=[0.0, 0.0, 0.025])

width = 500
height = 500
fov = 60
aspect = width / height
near = 0.02
far = 1

view_matrix = p.computeViewMatrix([0, 0, 0.5], [0, 0, 0], [1, 0, 0])
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)


# Get depth values using the OpenGL renderer
images = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
rgb_opengl = (np.reshape(images[2], (width, height, 4)))
rgbim_no_alpha = rgba2rgb(rgb_opengl)


# rgbim = Image.fromarray(rgb_opengl)
# rgbim_no_alpha = rgbim.convert('RGB')

plt.subplot(1, 2, 1)
plt.imshow(rgbim_no_alpha, vmin=0, vmax=1)
plt.title('OpenGL Renderer')


# depth_buffer_opengl = np.reshape(images[3], [width, height])
# depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)

# # Get depth values using Tiny renderer
# images = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_TINY_RENDERER)
# depth_buffer_tiny = np.reshape(images[3], [width, height])
# depth_tiny = far * near / (far - (far - near) * depth_buffer_tiny)

# p.disconnect()

# # Plot both images - should show depth values of 0.45 over the cube and 0.5 over the plane
# plt.subplot(1, 2, 1)
# plt.imshow(depth_opengl, cmap='gray', vmin=0, vmax=1)
# plt.title('OpenGL Renderer')

# plt.subplot(1, 2, 2)
# plt.imshow(depth_tiny, cmap='gray', vmin=0, vmax=1)
# plt.title('Tiny Renderer')

plt.show()