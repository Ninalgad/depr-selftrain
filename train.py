import argparse
import gc
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.optim import Adam

from loaders import get_static_loader
from datasets.depsign import DepSign2023
from evaluation import evaluate
from model import get_classifier
from utils import *

parser = argparse.ArgumentParser()
# dataset
parser.add_argument('--data_dir', type=str, default=".", help='directory to store the labeled dataset')

# training
parser.add_argument('--input_len', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--model_name', type=str, default='bert-base-uncased')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--num_epochs', type=int, default=10)

# others
parser.add_argument('--output_dir', type=str, default="./output/base")
parser.add_argument('--store_name', type=str, default="model")
parser.add_argument('--seed', type=int, default=0, help='Seed data split')
parser.add_argument('--debug', type=bool, default=False)

# checkpoints
parser.add_argument('--pretrained_algo', type=str, default=None)

args = parser.parse_args()


def main():
    if args.debug:
        logger.info("Running in debugging mode")
        args.batch_size = 2
        args.input_len = 8
        args.num_epochs = 1

    make_directories(args.output_dir)
    # create algorithm
    model, tokenizer = get_classifier(args.model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    logger.info(f"Created classifier using `{args.model_name}`")

    # load self-supervised pre-trained model and optimizer
    if args.pretrained_algo:
        checkpoint = torch.load(args.pretrained_algo, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint
        logger.info(f"Loaded pretrained weights found in '{args.pretrained_algo}'")

    # create dataset
    logger.info(f"Creating dataset in '{args.data_dir}'")
    dataset = DepSign2023(args.data_dir)
    train_df, dev_df, test_df = dataset.get_data_splits()

    # create loaders
    train_loader = get_static_loader(
        tokenizer, train_df['Text data'].tolist(), train_df.Label,
        max_length=args.input_len, batch_size=args.batch_size, shuffle=True)
    val_loader = get_static_loader(
        tokenizer, dev_df['Text data'].tolist(), dev_df.Label,
        max_length=args.input_len, batch_size=args.batch_size)
    test_loader = get_static_loader(
        tokenizer, test_df['Text data'].tolist(), test_df.Label.values,
        max_length=args.input_len, batch_size=args.batch_size)

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
    checkpoint_filename, dev_results_filename = "", ""
    criterion = torch.nn.CrossEntropyLoss()

    def train_step(inp):
        inp = {k: v.to(device) for k, v in inp.items()}

        labels = inp.pop('labels').float()

        logits = model(**inp).logits

        optimizer.zero_grad()
        loss_ = criterion(logits, labels)
        loss_.backward()
        optimizer.step()

        return loss_.item()

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = []
        gc.collect()
        for batch in train_loader:
            loss = train_step(batch)
            train_loss.append(loss)
            global_step += 1

            if args.debug:
                break

        gc.collect()
        with torch.no_grad():
            results = evaluate(model, val_loader, device, debug=args.debug)
        val = results['macro f1-score']
        train_loss = np.mean(train_loss)
        logger.info(f"Epoch: {epoch}, Train Loss: {train_loss:,.4f}, Dev f1-score: {val:,.4f}")

        if val > best_val:
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
