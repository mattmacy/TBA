import os
from argparse import ArgumentParser
from .train import train

def get_args():
    parser = ArgumentParser(description='PyTorch BIDAF model')
    # directories and data
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--data_cache', type=str, default=os.path.join(os.getcwd(), '.data_cache'))
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.42B')
    parser.add_argument('--resume_snapshot', type=str, default='')

    # train / test parameters
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--init_lr', type=float, default=0.5)
    parser.add_argument('--do_p', type=float, default=0.8)
    parser.add_argument('--wd', type=float, default=0.)
    parser.add_argument('--d_hidden', type=int, default=100)
    parser.add_argument('--d_embed', type=int, default=300)
    parser.add_argument('--d_char_embed', type=int, default=8)
    parser.add_argument('--d_char_out', type=int, default=100)
    parser.add_argument('--d_out_chan', type=string, default="100")
    parser.add_argument('--d_char_filter', type=string, default="5")

    #Advanced training options
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--logit_func', type=string, default="tri_linear")
    parser.add_argument('--answer_func', type=string, default="linear")
    
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
   train(args)

if __name__ == '__main__':
    main()
