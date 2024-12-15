import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import IMDBBertDataset
from model import BERT
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertTrainer:

    def __init__(self,
                 model: BERT,
                 dataset: IMDBBertDataset,
                 log_dir: Path,
                 checkpoint_dir: Path = None,
                 print_progress_every: int = 10,
                 print_accuracy_every: int = 50,
                 batch_size: int = 24,
                 learning_rate: float = 5e-3,
                 epochs: int = 5
                 ):
    
        self.model = model
        self.dataset = dataset
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.current_epoch = 0

        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.writer = SummaryWriter(str(log_dir))
        self.checkpoint_dir = checkpoint_dir

        self.mlm_criterion = nn.NLLLoss(ignore_index=0).to(device)
        self.nsp_criterion = nn.BCEWithLogitsLoss().to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.015)

        self._splitter_size = 35

        self._ds_len = len(self.dataset)
        self._batched_len = self._ds_len // self.batch_size

        self._print_every = print_progress_every
        self._accuracy_every = print_accuracy_every

    def print_summary(self):
        ds_len = len(self.dataset)

        print("Model Summary\n")
        print('=' * self._splitter_size)
        print(f"Device: {device}")
        print(f"Training dataset len: {ds_len}")
        print(f"Max / Optimal sentence len: {self.dataset.optimal_sentence_length}")
        print(f"Vocab size: {len(self.dataset.vocab)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Batched dataset len: {self._batched_len}")
        print('=' * self._splitter_size)
        print()
    
    def __call__(self):
        for self.current_epoch in range(self.current_epoch, self.epochs):
            loss = self.train(self.current_epoch)
            self.save_checkpoint(self.current_epoch, step=-1, loss=loss)
    
    def train(self, epoch: int):
        print(f"Begin epoch {epoch}")

        prev = time.time()
        average_nsp_loss = 0
        average_mlm_loss = 0
        for i, data in enumerate(self.loader):
            index = i + 1
            inp, mask, inverse_token_mask, token_target, nsp_target = data
            self.optimizer.zero_grad()

            # token -> (batch_size, seq_len, vocab_size), nsp -> (batch_size, 2)
            token, nsp = self.model(inp, mask)

            tm = inverse_token_mask.unsqueeze(-1).expand_as(token)
            token = token.masked_fill_(tm, 0)

            loss_mlm = self.mlm_criterion(token.transpose(1,2), token_target)
            loss_nsp = self.nsp_criterion(nsp, nsp_target)

            loss = loss_mlm + loss_nsp
            average_mlm_loss += loss_mlm
            average_nsp_loss += loss_nsp

            loss.backward()
            self.optimizer.step()

            if index % self._print_every == 0:
                elapsed = time.gmtime(time.time() - prev)
                s = self.training_summary(elapsed, index, average_mlm_loss, average_nsp_loss)

                if index % self._accuracy_every == 0:
                    s += self.accuracy_summary(index, token, nsp, token_target, nsp_target, inverse_token_mask)

                print(s)

                average_mlm_loss, average_nsp_loss = 0, 0
        return loss

    def training_summary(self, elaspsed, index, average_mlm_loss, average_nsp_loss):
        passed = percentage(self.batch_size, self._ds_len, index)
        global_step = self.current_epoch * len(self.loader) + index

        print_mlm_loss = average_mlm_loss / self._print_every
        print_nsp_loss = average_nsp_loss / self._print_every

        s = f"{time.strftime('%H:%M:%S', elaspsed)}"
        s += f" | Epoch {self.current_epoch + 1} | {index} / {self._batched_len} ({passed}%)"
        s += f" | MLM loss {print_mlm_loss: 6.2f} | NSP loss {print_nsp_loss: 6.2f}"

        self.writer.add_scalar("MLM loss", print_mlm_loss, global_step=global_step)
        self.writer.add_scalar("NSP loss", print_nsp_loss, global_step=global_step)

        return s
    
    def accuracy_summary(self, index, token, nsp, token_target, nsp_target, inverse_token_mask):
        global_step = self.current_epoch * len(self.loader) + index
        mlm_acc = mlm_accuracy(token, token_target, inverse_token_mask)
        nsp_acc = nsp_accuracy(nsp, nsp_target)

        self.writer.add_scalar("MLM train accuracy", mlm_acc, global_step=global_step)
        self.writer.add_scalar("NSP train accuracy", nsp_acc, global_step=global_step)

        return f" | MLM accuracy {mlm_acc} | NSP accuracy {nsp_acc}"
    
    def save_checkpoint(self, epoch, step, loss):
        if not self.checkpoint_dir:
            return

        prev = time.time()
        name = f"bert_epoch{epoch}_step{step}_{datetime.now(tz=timezone.utc).timestamp():.0f}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict,
            'loss': loss,
        }, self.checkpoint_dir.joinpath(name))

        print()
        print('=' * self._splitter_size)
        print(f"Model saved as '{name}' for {time.time() - prev:2f}s")
        print('=' * self._splitter_size)
        print()