import csv
import math
import numpy as np

# Function to generate traj1
def generate_traj1(num_points):
    # Linear increase and decrease in the z domain
    z_values = [i / (num_points - 1) for i in range(num_points // 2)]
    z_values += [1 for i in range(200)]
    z_values += [1.0 - i / (num_points - 1) for i in range((num_points // 2)-200)]
    
    # Linear movement in x, y plane until half of the data points
    xy_values = np.array([[0,0,0] for i in range(num_points)])
    xy_values = np.array([[i/1000, i/1000, 1.0] for i in range(num_points)])

    # Circular movement in x, y plane for the remaining data points
    circular_radius = 0.3
    theta_values = [i * 2 * math.pi / (num_points // 2 - 1) for i in range(num_points // 2)]
    circular_xy = np.array([[circular_radius * math.cos(theta), circular_radius * math.sin(theta), 1.0] for theta in theta_values])
    circular_xy_value = np.array([[0,0,0] for i in range(num_points//2)])
    circular_xy = np.concatenate((circular_xy_value,circular_xy),axis=0)

    traj1_positions = circular_xy + xy_values
    traj1_positions = [[point[0], point[1], z] for point, z in zip(traj1_positions, z_values)]

    return traj1_positions

# Function to generate traj2
def generate_traj2(num_points):
    # Linear increase and decrease in the z domain
    z_values = [1+ i/(num_points - 1) for i in range(num_points // 2)]
    z_values += [1+ 1.0 - i/(num_points - 1) for i in range(num_points // 2)]
    
    # Linear movement in x, y plane until half of the data points
    xy_values = np.array([[0,0,0] for i in range(num_points)])
    xy_values = np.array([[i/1000, i/1000, 1.0] for i in range(num_points)])

    # Circular movement in x, y plane for the remaining data points
    traj2_positions = xy_values
    traj2_positions = [[point[0], point[1], z] for point, z in zip(traj1_positions, z_values)]

    return traj2_positions

def generate_traj3(num_points):
    # Linear increase and decrease in the z domain
    z_values = [1+ i/(num_points - 1) for i in range(num_points // 2)]
    z_values += [1+ 1.0 - i/(num_points - 1) for i in range(num_points // 2)]
    
    # Linear movement in x, y plane until half of the data points
    xy_values = np.array([[0,0,0] for i in range(num_points)])
    xy_values = np.array([[i/1000, i/1000, 1.0] for i in range(num_points)])

    # Circular movement in x, y plane for the remaining data points
    circular_radius = 0.3
    theta_values = [i * 2 * math.pi / (num_points // 2 - 1) for i in range(num_points // 2)]
    circular_xy = np.array([[circular_radius * math.cos(5*theta), circular_radius * math.sin(5*theta), 1.0] for theta in theta_values])
    circular_xy_value = np.array([[0,0,0] for i in range(num_points//2)])
    circular_xy = np.concatenate((circular_xy_value,circular_xy),axis=0)

    traj3_positions = circular_xy + xy_values
    traj3_positions = [[point[0], point[1], z] for point, z in zip(traj1_positions, z_values)]

    return traj3_positions

def generate_traj4(num_points):
    # Linear increase and decrease in the z domain
    z_values = [1-(i / (num_points - 1)) for i in range(num_points)]

    # Linear movement in x, y plane until half of the data points
    xy_values = np.array([[0,0,0] for i in range(num_points)])
    xy_values = np.array([[i/1000, i/1000, 1.0] for i in range(num_points)])

    # Circular movement in x, y plane for the remaining data points
    circular_radius = 0.3
    theta_values = [i * 2 * math.pi / (num_points // 2 - 1) for i in range(num_points)]
    circular_xy = np.array([[circular_radius * math.cos(4*theta), circular_radius * math.sin(4*theta), 1.0] for theta in theta_values])

    traj1_positions = circular_xy + xy_values
    traj1_positions = [[point[0], point[1], z] for point, z in zip(traj1_positions, z_values)]

    return traj1_positions


def generate_traj5(num_points):
    # Linear increase and decrease in the z domain
    z_values = [1-(i / (num_points - 1)) for i in range(num_points)]

    # Linear movement in x, y plane until half of the data points
    xy_values = np.array([[0,0,0] for i in range(num_points)])
    xy_values = np.array([[i/1000, i/1000, 1.0] for i in range(num_points)])

    traj1_positions = xy_values
    traj1_positions = [[point[0], point[1], z] for point, z in zip(traj1_positions, z_values)]

    return traj1_positions

def generate_traj6(num_points):
    # Linear increase and decrease in the z domain
    z_values = [1-(i / (num_points - 1)) for i in range(num_points)]

    # Linear movement in x, y plane until half of the data points
    xy_values = np.array([[0,0,0] for i in range(num_points)])
    xy_values = np.array([[i/1000, i/1000, 1.0] for i in range(num_points//2)])
    xy_values = np.array([[0.5-i/500, 0.5-i/500, 1.0] for i in range(num_points//2)])

    traj1_positions = xy_values
    traj1_positions = [[point[0], point[1], z] for point, z in zip(traj1_positions, z_values)]

    return traj1_positions

def generate_traj7(num_points):
    # Linear increase and decrease in the z domain
    z_values = [1-(2*i / (num_points - 1)) for i in range(num_points//4)]
    z_values += [0.5+(i / (2*(num_points - 1))) for i in range(num_points//4)]
    z_values += [0.75-1.5*(i / (num_points - 1)) for i in range(num_points//2)]
    
    # Linear movement in x, y plane until half of the data points
    xy_values = np.array([[0,0,0] for i in range(num_points)])
    xy_values = np.array([[i/1000, i/1000, 1.0] for i in range(num_points//2)])
    xy_values += np.array([[0.5-i/800, 0.5-i/800, 1.0] for i in range(num_points//2)])

    traj1_positions = xy_values
    traj1_positions = [[point[0], point[1], z] for point, z in zip(traj1_positions, z_values)]

    return traj1_positions

def generate_traj8(num_points):
    # Linear increase and decrease in the z domain
    z_values = [1-(2*i / (num_points - 1)) for i in range(num_points//4)]
    z_values += [0.5+(i / (2*(num_points - 1))) for i in range(num_points//4)]
    z_values += [0.75-1.5*(i / (num_points - 1)) for i in range(num_points//2)]
    
    # Linear movement in x, y plane until half of the data points
    xy_values = np.array([[0,0,0] for i in range(num_points)])
    xy_values = np.array([[i/1000, i/1000, 1.0] for i in range(num_points)])

    circular_radius = 0.3
    theta_values = [i * 2 * math.pi / (num_points // 2 - 1) for i in range(num_points)]
    circular_xy = np.array([[circular_radius * math.cos(4*theta), circular_radius * math.sin(4*theta), 1.0] for theta in theta_values])

    traj1_positions = circular_xy + xy_values
    traj1_positions = [[point[0], point[1], z] for point, z in zip(traj1_positions, z_values)]

    return traj1_positions


# Specify the number of data points you want
num_data_points = 1000

# Generate traj1 data
traj1_positions = generate_traj4(num_data_points)

# Generate traj2 data
traj2_positions = generate_traj5(num_data_points)

traj3_positions = generate_traj6(num_data_points)

traj4_positions = generate_traj7(num_data_points)

traj5_positions = generate_traj8(num_data_points)


# Specify the CSV file paths
traj1_csv_file = 'circular.csv'
traj2_csv_file = 'linear.csv'
traj3_csv_file = 'somewhere.csv'
traj4_csv_file = 'up_down.csv'
traj5_csv_file = 'circular_updown.csv'

# Write traj1 data to CSV file
with open(traj1_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write traj1 data
    writer.writerows(traj1_positions)

print(f'Traj1 CSV file created successfully at {traj1_csv_file}')

# Write traj2 data to CSV file
with open(traj2_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write traj2 data
    writer.writerows(traj2_positions)

print(f'Traj2 CSV file created successfully at {traj2_csv_file}')

with open(traj3_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write traj3 data
    writer.writerows(traj3_positions)

print(f'Traj2 CSV file created successfully at {traj3_csv_file}')

# Write traj1 data to CSV file
with open(traj4_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write traj1 data
    writer.writerows(traj4_positions)

print(f'Traj1 CSV file created successfully at {traj4_csv_file}')

# Write traj1 data to CSV file
with open(traj5_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write traj1 data
    writer.writerows(traj5_positions)

print(f'Traj1 CSV file created successfully at {traj5_csv_file}')
