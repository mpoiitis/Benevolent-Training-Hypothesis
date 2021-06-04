import steadylearner
from steadylearner.utils import parse_args
from steadylearner.experiments import run_cnn, run_mlp, run_mnist_exp
from tqdm import tqdm

def run_experiment():
    args = parse_args()
    for i in tqdm(range(args.repeats)):
        if args.cnn:
            run_cnn()
        else:
            run_mlp()

        # run_mnist_exp()