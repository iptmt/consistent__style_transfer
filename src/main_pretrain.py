import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

from model.mlm import MLM
from model.match import Matcher
from model.classifier import TextCNN

from wmd import WMDdistance
from vocab import BPETokenizer
from loader import StyleDataset, load_s2l, collate_pretrain 

STAGE = "pretrain"

class PretrainModel(pl.LightningModule):
    def __init__(self, args):
        super(PretrainModel, self).__init__()

        self.hparams = args

        self.w2v = WMDdistance.load(f"{args.dump_dir}/{args.dataset}/{args.dataset}-w2v.bin")
        self.vocab = BPETokenizer.load(f"{args.dump_dir}/{args.dataset}/{args.dataset}-vocab.json",
                                       f"{args.dump_dir}/{args.dataset}/{args.dataset}-merges.txt")

        self.classifier = TextCNN(len(self.vocab), n_class=args.n_class)
        self.classifier.load_state_dict(torch.load(self.hparams.task_dump_dir + "cls.pth"))
        self.matcher = Matcher(len(self.vocab))
        self.matcher.load_state_dict(torch.load(self.hparams.task_dump_dir + "mat.pth"))
        self.denoiser = MLM(len(self.vocab), n_class=args.n_class)
        self.denoiser.load_state_dict(torch.load(self.hparams.task_dump_dir + "dn.pth"))

        self.data_dir = f"{args.data_dir}/{args.dataset}"

        self.ce_crit = nn.CrossEntropyLoss()
        self.mse_crit = nn.MSELoss()

        self.flags = {"cls": True, "mat": True, "dn": True}
        self.named_models = {"cls": self.classifier, "mat": self.matcher, "dn": self.denoiser}
        self.best_eval = {name: float("inf") if self.flags[name] else 0. for name in self.flags}
                

    def forward(self, x, noise_x_1, noise_x_2, noise_x, label):
        # classification
        s_logits = self.classifier(x) if self.flags["cls"] else None

        # content matching
        c_logits = self.matcher(noise_x_1, noise_x_2) if self.flags["mat"] else None

        # denoising
        dn_logits = self.denoiser(noise_x) if self.flags["dn"] else None

        return s_logits, c_logits, dn_logits
    
    def configure_optimizers(self):
        params = list(self.classifier.parameters()) + list(self.matcher.parameters()) + list(self.denoiser.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-4)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, nx_1, nx_2, nx, label, c_label = batch

        s_logits, c_logits, dn_logits = self.forward(x, nx_1, nx_2, nx, label)

        s_loss = self.ce_crit(s_logits, label) if s_logits is not None else 0.
        c_loss = self.mse_crit(c_logits, c_label) if c_logits is not None else 0.
        dn_loss = self.ce_crit(dn_logits.reshape(-1, dn_logits.size(-1)), x.reshape(-1)) if dn_logits is not None else 0.

        log_info = {"s_loss": s_loss, "c_loss": c_loss, "dn_loss": dn_loss}

        return {'loss': s_loss + c_loss + dn_loss, 'progress_bar': log_info, 'log': log_info}
    
    def validation_step(self, batch, batch_idx):
        x, nx_1, nx_2, nx, label, c_label = batch

        s_logits, c_logits, dn_logits = self.forward(x, nx_1, nx_2, nx, label)

        s_loss = self.ce_crit(s_logits, label) if s_logits is not None else 0.
        c_loss = self.mse_crit(c_logits, c_label) if c_logits is not None else 0.
        dn_loss = self.ce_crit(dn_logits.reshape(-1, dn_logits.size(-1)), x.reshape(-1)) if dn_logits is not None else 0.
        
        return {
            's_loss': s_loss, "c_loss": c_loss, "dn_loss": dn_loss
        }
    
    def validation_end(self, outputs):
        s_loss, c_loss, dn_loss = [], [], []
        for o in outputs:
            s_loss.append(o["s_loss"])
            c_loss.append(o["c_loss"])
            dn_loss.append(o["dn_loss"])
        s_loss, c_loss, dn_loss = sum(s_loss)/len(s_loss), sum(c_loss)/len(c_loss), sum(dn_loss)/len(dn_loss)
        for name, loss in [("cls", s_loss), ("mat", c_loss), ("dn", dn_loss)]:
            if self.flags[name]:
                if self.best_eval[name] < loss:
                    self.flags[name] = False
                else:
                    self.best_eval[name] = loss
                    torch.save(self.named_models[name].state_dict(), f"{self.hparams.task_dump_dir}/{name}.pth")
        val_loss = sum(list(self.best_eval.values()))
        print(f"CLS: {self.flags['cls']}-{self.best_eval['cls']}\n" + \
              f"MAT: {self.flags['mat']}-{self.best_eval['mat']}\n" + \
              f"DN: {self.flags['dn']}-{self.best_eval['dn']}\n" +\
              f"val_loss: {val_loss}")
        return {
            "progress_bar": {"val_loss": val_loss},
            "log": {"val_loss": val_loss}
        }
    
    @pl.data_loader
    def train_dataloader(self):
        dataset = StyleDataset([f"{self.data_dir}/style.train.0", f"{self.data_dir}/style.train.1"], self.vocab,
                                   max_len=self.hparams.max_len, load_func=load_s2l)
        data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, 
                                 collate_fn=collate_pretrain(self.vocab, self.w2v))
        return data_loader
    
    @pl.data_loader
    def val_dataloader(self):
        dataset = StyleDataset([f"{self.data_dir}/style.dev.0", f"{self.data_dir}/style.dev.1"], self.vocab, 
                                   max_len=self.hparams.max_len, load_func=load_s2l)
        data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, 
                                 collate_fn=collate_pretrain(self.vocab, self.w2v))
        return data_loader

def construct_trainer(args):
    logger = TensorBoardLogger(save_dir=args.log_dir,
                               name=STAGE)
    early_stop = EarlyStopping(monitor="val_loss",
                               patience=1,
                               mode="min")
    trainer = Trainer(logger=logger,
                      gradient_clip_val=5.0,
                      checkpoint_callback=False,
                      early_stop_callback=early_stop,
                      max_epochs=args.epochs,
                      gpus=args.device)
    return trainer


if __name__ == "__main__":
    from arguments import fetch_args
    args = fetch_args()

    if not os.path.exists(f"{args.dump_dir}/{args.dataset}/{STAGE}"):
        os.mkdir(f"{args.dump_dir}/{args.dataset}/{STAGE}")
    args.task_dump_dir = f"{args.dump_dir}/{args.dataset}/{STAGE}"
    args.log_dir = f"{args.log_dir}/{args.dataset}"

    model = PretrainModel(args)
    trainer = construct_trainer(args)
    trainer.fit(model)