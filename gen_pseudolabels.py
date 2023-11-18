import os
import torch
import argparse
from loguru import logger
import numpy as np

from loaders import get_static_loader
from datasets.rmhd import RedditMentalHealthDataset
from utils import make_directories
from model import get_classifier


parser = argparse.ArgumentParser(description='Apply standard trained model to generate labels on unlabeled data')

parser.add_argument('--dataset', default='rmhd-small', choices=['rmhd', 'rmhd-small'])
parser.add_argument('--output_dir', type=str, default=".")

# load trained models
parser.add_argument('--model_name', type=str, default='bert-base-uncased')
parser.add_argument('--trained_model', type=str)

# data related
parser.add_argument('--unlabeled_data_dir', default='./unlabeled', type=str,
                    help='directory that has unlabeled data')

# prediction
parser.add_argument('--input_len', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--debug', type=bool, default=False)
args = parser.parse_args()

make_directories(args.output_dir)
if args.debug:
    logger.info("Running in debugging mode")
    args.batch_size = 2
    args.input_len = 8

dataset_name = args.dataset
is_small = False
if 'small' in dataset_name:
    is_small = True
dataset = RedditMentalHealthDataset(args.unlabeled_data_dir, is_small)

# download dataset
logger.info(f"Creating `{dataset_name}` dataset")
dataset.download()
unlabeled_texts = dataset.get_texts()
if args.debug:
    unlabeled_texts = unlabeled_texts[:10]

# create algorithm
model, tokenizer = get_classifier(args.model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
logger.info(f"Created classifier using `{args.model_name}`")

# load trained model
model.load_state_dict(torch.load(args.trained_model, map_location=device)['model_state_dict'])
logger.info(f"Loaded pretrained weights found in '{args.trained_model}'")

# data generator
pred_loader = get_static_loader(
    tokenizer, unlabeled_texts,
    max_length=args.input_len, batch_size=args.batch_size
)

# predict
predictions = np.array([], 'uint8')
logger.info(f"Predicting for {len(unlabeled_texts)} samples")
for batch in pred_loader:
    batch = {k: v.to(device) for k, v in batch.items()}

    logits = model(**batch).logits

    p = logits.detach().cpu().numpy()
    p = np.argmax(p, axis=-1)

    predictions = np.append(predictions, p)

# save predictions
save_filename = os.path.join(args.output_dir, f'{dataset_name}-predictions.npy')
np.save(save_filename, predictions)
logger.success(f"Completed, model saved predictions to {save_filename}")
