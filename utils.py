import os
import json
import torch
import pathlib


def save_state(args, state, is_best):
    filename = os.path.join(args.output_dir, args.store_name)
    suffix = ".pt"
    if is_best:
        suffix = "-best.pt"
    filename += suffix
    torch.save(state, filename)
    return filename


def save_results(args, res, split):
    filename = os.path.join(args.output_dir, f'{split}_results.json')
    with open(filename, 'w') as f:
        json.dump(res, f)
    return filename


def make_directories(dir_name):
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
