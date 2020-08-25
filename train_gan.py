import argparse

from util.dataset import Dataset
from model.custoom_gan import CGAN
import configuration

dataset_name = 'unimib'
latent_dim = 50
cond_dim = 50


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_name', '--dataset_name', type=str,
                        default='unimib', help='name dataste for gan train')
    parser.add_argument('-latend_dim', '--latent_dim', type=int,
                        default=50, help='dimension of latent space (input generator)')
    parser.add_argument('-cond_dim', '--cond_dim', type=int,
                        default=50, help='dimension conditional information to gan')
    parser.add_argument('-magnitude', '--magnitude', type=bool,
                        default=True, help='use magnitude or not')
    parser.add_argument('-win_len', '--win_len', type=int,
                        default=100, help='len of window')
    parser.add_argument('-test_folder', '--step', type=list,
                        default=[0], help='list of folder to use as test')
    parser.add_argument('-outer_dir', '--outer_dir', type=str, default='OuterPartition_magnitude_5.0/', help='dir with all subdir for every folder')
    args = parser.parse_args()

    conf = configuration.config[dataset_name]

    if args.magnitude:
        channel = conf['AXIS'] + conf['AXIS_MAGNITUDE']
    else:
        channel = conf['AXIS']

    # define dataset
    dataset = Dataset(path=conf['PATH_OUTER_PARTITION'],
                      name=args.dataset_name,
                      channel=channel,
                      winlen=args.win_len,
                      user_num=conf['NUM_CLASSES_USER'],
                      act_num=conf['NUM_CLASSES_ACTIVITY'],
                      outer_dir=args.outer_dir,
                      )

    # load data
    train_data, train_la, train_lu, _, _, _ = dataset.load_data(
        step=args.step, magnitude=args.magnitude)

    # define GAN
    cgan = CGAN(latent_dim, cond_dim)
    cgan.new_generator()
    cgan.new_discriminator()

    # train GAN
    cgan.train_cgan()
