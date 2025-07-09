from argparse import ArgumentParser

def str2bool(x):
    assert x == "True" or x == "False"
    return True if x == "True" else False

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    
    parser.add_argument("--out-path", type=str, default=None,)
    parser.add_argument("--ckpt-path", type=str, default=None,)
    parser.add_argument("--snapshot-path", type=str, default=None,)
    parser.add_argument("--data-base-path", type=str, default=None,)
    parser.add_argument("--regen-times", type=int, default=16,)

    parser.add_argument("--batch-num", type=int, default=1,)
    parser.add_argument("--batch-size", type=int, default=100,)
    parser.add_argument("--batch-size-per-gpu", type=int, default=100,)
    parser.add_argument("--eval-batch-num", type=int, default=1,)
    parser.add_argument("--eval-batch-size", type=int, default=100,)
    parser.add_argument("--eval-batch-size-per-gpu", type=int, default=100,)
    parser.add_argument("--problem-batch-num", type=int, default=1,)
    parser.add_argument("--problem-batch-size-per-gpu", type=int, default=100,)
    parser.add_argument("--total-problem-num", type=int, default=100,)
    parser.add_argument("--check-loss", type=str2bool, default=False)
    
    return parser.parse_args()