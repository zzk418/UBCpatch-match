import argparse
import torch
from new_model import *
from data import train_loader, vaild_loader
from train_fp16 import Trainer

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning Rate')
parser.add_argument('--steps', '-n', default=200, type=int, help='No of Steps')
parser.add_argument('--gpu', '-p', default=True, help='Train on GPU')
parser.add_argument(
    '--fp16', default=True, help='Train with FP16 weights')
parser.add_argument(
    '--loss_scaling', '-s', default=True, help='Scale FP16 losses')
# parser.add_argument(
#     '--model', '-m', default='resnet50', type=str, help='Name of Network')
args = parser.parse_args()


if args.gpu and torch.cuda.is_available():
    train_on_gpu = True
    # CuDNN must be enabled for FP16 training.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

# model_name = args.model
model_name = 'matchnet'
model = MatchNet(new_module)
model.load_state_dict(torch.load('outmatchnet-weights-last.pt'), strict=False)
model = model.cuda()

if __name__ == '__main__':
    trainer = Trainer(model_name, model, args.lr, train_on_gpu, args.fp16,
                  args.loss_scaling)
    trainer.train_and_evaluate(train_loader, vaild_loader, args.steps)
