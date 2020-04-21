import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

# from model.mlm import MLM
from model.rnn import DenoiseLSTM

from vocab import BPETokenizer
from loader import StyleDataset, load_s2l, collate_warmup

STAGE = "warmup"

class WarmupModel(pl.LightningModule):
    def __init__(self, args):
        super(WarmupModel, self).__init__()
        self.hparams = args

        self.vocab = BPETokenizer.load(f"{args.dump_dir}/{args.dataset}/{args.dataset}-vocab.json",
                                       f"{args.dump_dir}/{args.dataset}/{args.dataset}-merges.txt")

        self.generator = DenoiseLSTM(len(self.vocab), args.n_class, args.max_len)

        self.data_dir = f"{args.data_dir}/{args.dataset}"

        self.criterion = nn.CrossEntropyLoss()

        self.best_eval = float("inf")
        self.max_iter = -1
 
    def forward(self, nx, n_labels, x, labels):
        dn_logits = self.generator(nx, n_labels, x, labels)
        return dn_logits
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        nx, x, labels = batch
        w = 1 - self.global_step / self.max_iter

        dn_logits = self.forward(nx, labels, x, labels)
        with torch.no_grad():
            tokens_tsf = self.forward(x, labels, None, 1 - labels).argmax(-1)
        bk_logits = self.forward(tokens_tsf, 1 - labels, x, labels)
        dn_loss = self.criterion(dn_logits.reshape(-1, dn_logits.size(-1)), x.reshape(-1))
        bk_loss = self.criterion(bk_logits.reshape(-1, bk_logits.size(-1)), x.reshape(-1))

        loginfo = {"dn_loss": dn_loss, "bk_loss": bk_loss}
        return {"loss": w * dn_loss + bk_loss, "progress_bar": loginfo, "log": loginfo}
    
    def validation_step(self, batch, batch_idx):
        nx, x, labels = batch
        dn_logits = self.forward(nx, labels, x, labels)
        with torch.no_grad():
            tokens_tsf = self.forward(x, labels, None, 1 - labels).argmax(-1)
        bk_logits = self.forward(tokens_tsf, 1 - labels, x, labels)

        dn_loss = self.criterion(dn_logits.reshape(-1, dn_logits.size(-1)), x.reshape(-1))
        bk_loss = self.criterion(bk_logits.reshape(-1, bk_logits.size(-1)), x.reshape(-1))
        return {"loss": dn_loss.item() + bk_loss.item()}
    
    def validation_end(self, outputs):
        losses = np.array([o["loss"] for o in outputs])
        loss = losses.mean()
        if self.best_eval > loss:
            self.best_eval = loss
            torch.save(self.generator.state_dict(), f"{self.hparams.task_dump_dir}/G.pth")
        return {"progress_bar": {"loss": loss}, "log": {"val_loss": loss}}
    
    @pl.data_loader
    def train_dataloader(self):
        dataset = StyleDataset([f"{self.data_dir}/style.train.0", f"{self.data_dir}/style.train.1"], self.vocab, 
                                max_len=self.hparams.max_len, load_func=load_s2l)
        data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, 
                                 collate_fn=collate_warmup)
        self.max_iter = len(data_loader) * self.hparams.epochs
        return data_loader
    
    @pl.data_loader
    def val_dataloader(self):
        dataset = StyleDataset([f"{self.data_dir}/style.dev.0", f"{self.data_dir}/style.dev.1"], self.vocab, 
                                max_len=self.hparams.max_len, load_func=load_s2l)
        data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, 
                                 collate_fn=collate_warmup)
        return data_loader
    
def construct_trainer(args):
    logger = TensorBoardLogger(save_dir=args.log_dir,
                               name=STAGE)
    early_stop = EarlyStopping(monitor="val_loss",
                               patience=1,
                               mode="min")
    trainer = Trainer(logger=logger,
                      gradient_clip_val=1.0,
                      checkpoint_callback=False,
                      early_stop_callback=early_stop,
                      max_epochs=args.epochs,
                      gpus=args.device)
    return trainer


if __name__ == "__main__":
    from arguments import fetch_args
    args = fetch_args()

    if args.dataset == "yelp":
        args.epochs = 1
        args.batch_size = 512
    elif args.dataset == "book":
        args.epochs = 1
        args.batch_size = 512
    else:
        raise ValueError

    if not os.path.exists(f"{args.dump_dir}/{args.dataset}/{STAGE}"):
        os.mkdir(f"{args.dump_dir}/{args.dataset}/{STAGE}")
    args.task_dump_dir = f"{args.dump_dir}/{args.dataset}/{STAGE}"
    args.log_dir = f"{args.log_dir}/{args.dataset}"

    model = WarmupModel(args)
    trainer = construct_trainer(args)
    trainer.fit(model)