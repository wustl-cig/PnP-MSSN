import argparse

parser = argparse.ArgumentParser(description="MSSN")
# Settings of PnP
parser.add_argument("--data_name", type=str, default='MRI_Knee_58.mat', help='the filename of the data')
parser.add_argument("--gpu_id", type=str, default='0', help="Training batch size")
parser.add_argument("--num_iter", type=int, default=200, help="number of iterations in plug-and-play algorithm")

# Settings of MSSN
parser.add_argument("--batch_size", type=int, default=1, help="batch size of neural network, 1 for testing in PnP")
parser.add_argument("--patch_size", type=int, default=42, help="patch size of MSSN")
parser.add_argument("--state_num", type=int, default=8, help="Number of recurrent states in model")

parser.add_argument("--model_name", type=str, default='mssn', help="model name for module import")
parser.add_argument("--model_checkpoints", type=str, default='models/checkpoints/mssn-550000iters', help="path of model checkpoints")

parser.add_argument("--key_dim", type=int, default=128, help="channels of key in multi-head attention of MSSN")
parser.add_argument("--value_dim", type=int, default=128, help="channels of value in multi-head attention of MSSN")
parser.add_argument("--num_heads", type=int, default=2, help="number of heads in multi-head attention")
parser.add_argument("--sigma", type=int, default=5, help="noise level of the denoiser is trained on")

opt = parser.parse_args()