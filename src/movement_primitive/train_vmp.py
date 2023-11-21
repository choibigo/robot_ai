from vmp import VMP
import numpy as np
import argparse
import os
import glob

traj_path = "/workspace/data/traj_data"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train VMP from data/traj_data/(style)/traj*.csv and name it (style)')
    parser.add_argument('--style', required=True,
                        help='the name of motion style to train: need to have traj files data/traj/(style)/traj*.csv')
    args = parser.parse_args()

    style_traj_path = os.path.join(traj_path,args.style)
    traj_files = glob.glob(os.path.join(style_traj_path,'*.csv'))
    trajs = np.array([np.loadtxt(f, delimiter=',') for f in traj_files])
    print(trajs.shape)


    vmp = VMP(3, kernel_num=50, elementary_type='linear', use_out_of_range_kernel=False)
    linear_traj_raw = vmp.train(trajs[[0]])
    vmp.save_weights_to_file(args.style)