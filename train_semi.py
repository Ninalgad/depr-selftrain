import torch
import argparse
from loguru import logger
import numpy as np
import gc
from torch.optim import Adam

from evaluation import evaluate
from utils import *
from loaders import get_static_loader
from datasets.rmhd import RedditMentalHealthDataset
from datasets.depsign import DepSign2023
from utils import make_directories
from model import get_classifier


parser = argparse.ArgumentParser()

# data related
parser.add_argument('--dataset', default='rmhd-small', choices=['rmhd', 'rmhd-small'])
parser.add_argument('--pseudolabels', type=str, help='directory to store the labeled dataset')
parser.add_argument('--data_dir', type=str, default=".", help='directory to store the labeled dataset')
parser.add_argument('--unlabeled_data_dir', default='./unlabeled', type=str,
                    help='directory that has unlabeled data')

# training
parser.add_argument('--input_len', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--model_name', type=str, default='bert-base-uncased')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--steps_per_epoch', type=int, default=None)

# others
parser.add_argument('--output_dir', type=str, default="./output/semi")
parser.add_argument('--log_freq', type=int, default=1000)
parser.add_argument('--store_name', type=str, default="model")
parser.add_argument('--seed', type=int, default=0, help='Seed data split')
parser.add_argument('--steps', type=int, default=None)
parser.add_argument('--debug', type=bool, default=False)

args = parser.parse_args()


def main():
    make_directories(args.output_dir)
    if args.debug:
        logger.info("Running in debugging mode")
        args.batch_size = 2
        args.input_len = 8

    dataset_name = args.dataset
    is_small = False
    if 'small' in dataset_name:
        is_small = True

    # load unlabeled dataset
    dataset = RedditMentalHealthDataset(args.unlabeled_data_dir, is_small)
    # download dataset
    logger.info(f"Creating training `{dataset_name}` dataset")
    dataset.download()
    unlabeled_texts = dataset.get_texts()
    if args.debug:
        unlabeled_texts = unlabeled_texts[:10]

    logger.info(f"Loading pseudo labels from '{args.pseudolabels}'")
    labels = np.load(args.pseudolabels)
    assert len(labels) == len(unlabeled_texts), f"'{args.pseudolabels}' is not compatible with {dataset_name}"

    # create training algorithm
    model, tokenizer = get_classifier(args.model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    logger.info(f"Created classifier using `{args.model_name}`")

    # use labeled dataset for validation
    logger.info(f"Creating dataset in '{args.data_dir}'")
    dataset = DepSign2023(args.data_dir)
    _, dev_df, test_df = dataset.get_data_splits()

    # create loaders
    logger.info(f"Creating loaders for {len(labels)} samples")
    train_loader = get_static_loader(
        tokenizer, unlabeled_texts, labels,
        max_length=args.input_len, batch_size=args.batch_size, shuffle=True)
    val_loader = get_static_loader(
        tokenizer, dev_df['Text data'].tolist(), dev_df['Label'],
        max_length=args.input_len, batch_size=args.batch_size)
    test_loader = get_static_loader(
        tokenizer, test_df['Text data'].tolist(), test_df['Label'],
        max_length=args.input_len, batch_size=args.batch_size)

    # clear memory
    del labels, unlabeled_texts, dev_df, test_df
    gc.collect()

    # train
    logger.info(f"Starting training for {args.num_epochs} epochs")
    ckpt_filename, dev_res_filename = train(model, device, optimizer, train_loader, val_loader, args, logger)
    logger.info(f"Saved best model to '{ckpt_filename}'")
    logger.info(f"Saved `dev` results to '{dev_res_filename}'")

    # evaluate best model on test set
    logger.info("Evaluating on `test` set")
    model.load_state_dict(torch.load(ckpt_filename)['model_state_dict'])
    test_results = evaluate(model, test_loader, device, args.debug)

    # save test results
    results_filename = save_results(args, test_results, 'test')
    logger.success(f"Completed, model saved `test` results to {results_filename}")


def train(model, device, optimizer, train_loader, val_loader, args, logger):
    best_val, global_step = -np.inf, 0
    checkpoint_filename, dev_results_filename = None, None
    criterion = torch.nn.CrossEntropyLoss()

    def train_step(inp):
        inp = {k: v.to(device) for k, v in inp.items()}
        optimizer.zero_grad()
        labels = inp.pop('labels').float()
        outputs = model(**inp).logits
        loss_ = criterion(outputs, labels)
        loss_.backward()
        optimizer.step()
        return loss_.item()

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = []
        gc.collect()
        for i, batch in enumerate(train_loader):
            loss = train_step(batch)
            train_loss.append(loss)
            global_step += 1

            if ((i + 1) % args.log_freq) == 0:
                logger.info(f"Epoch: {global_step / len(train_loader):,.4f}, Train loss {np.mean(train_loss):,.4f}")

            if args.debug:
                break

            if (args.steps_per_epoch is not None) and (args.steps_per_epoch < i):
                break

        gc.collect()
        with torch.no_grad():
            results = evaluate(model, val_loader, device, debug=args.debug)
        val = results['macro f1-score']

        if val > best_val:
            logger.info(f"Epoch: {epoch}, New best f1 score: {val:,.4f} over {best_val:,.4f}")
            best_val = val
            dev_results_filename = save_results(args, results, 'dev')
            state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'f1-score': val
            }
            checkpoint_filename = save_state(args, state, True)
    return checkpoint_filename, dev_results_filename


if __name__ == "__main__":
    main()
