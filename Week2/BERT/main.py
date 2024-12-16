from datetime import datetime, timezone

import torch

from pathlib import Path

from dataset import IMDBBertDataset
from model import BERT
from trainer import BertTrainer


BASE_dir = Path(__file__).resolve().parent

EMB_size = 64
HIDDEN_size = 36
LEARNING_rate = 7e-5
EPOCHS = 15
BATCH_size = 12
NUM_heads = 4

CHECKPOINT_dir = BASE_dir.joinpath('data/bert_checkpoints')

timestamp = datetime.now(tz=timezone.utc).timestamp()
LOG_dir = BASE_dir.joinpath(f'data/logs/bert_experiment_{timestamp}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

if __name__=="__main__":
    print("Prepare dataset")
    ds = IMDBBertDataset(BASE_dir.joinpath('data/IMDB Dataset.csv'), ds_from=0, ds_to=50000)

    bert = BERT(
        len(ds.vocab),
        EMB_size,
        HIDDEN_size,
        NUM_heads
    ).to(device)

    trainer = BertTrainer(
        model=bert,
        dataset=ds,
        log_dir=LOG_dir,
        checkpoint_dir=CHECKPOINT_dir,
        print_progress_every=20,
        print_accuracy_every=50,
        batch_size=BATCH_size,
        learning_rate=LEARNING_rate,
        epochs=EPOCHS
    )

    trainer.print_summary()
    trainer()