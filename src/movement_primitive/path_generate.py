import argparse
import sys
sys.path.append('/workspace/src/')

from simulation.my_env import SimulationEnvironment
from movement_primitive.vmp import VMP
import numpy as np
import csv

def parse_args():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--style', required=True, help='the name of motion style to train or reproduce')       
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Build Enviroment
    env = SimulationEnvironment()
    pybullet_simulation = env.build()

    for i in range(1000):
        env.get_eff()
        print(env.end)
    traj1_csv_file = f'/workspace/data/traj_data/{args.style}/main_test_shaking.csv'

    # Write traj1 data to CSV file
    with open(traj1_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write traj1 data
        writer.writerows(env.end)
    print('success')
    

