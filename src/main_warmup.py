import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model.generator import RNNSearch

from vocab import Wrapped_ALBERT_Vocab
from loader import StyleDataset, load_s2l, collate_s2s_noise

STAGE = "warmup"

class CoarseTransfer(pl.LightningModule):
    def __init__(self, args):
        super(CoarseTransfer, self).__init__()
        self.args = args
        self.hparams = args

        self.vocab = Wrapped_ALBERT_Vocab(args.dump_dir)
        self.generator = RNNSearch(args.n_vocab, args.d_embed, args.d_enc_hidden, args.d_dec_hidden,
                                   args.n_enc_layer, args.n_dec_layer, args.n_class, args.p_drop, args.max_len)
        self.data_dir = f"{args.data_dir}/{args.dataset}"

        self.criterion = nn.CrossEntropyLoss()

        self.softmax = nn.Softmax(-1)

        self.tau = args.tau
        self.mle = True
 
    def forward(self, nx, labels, x, max_len, mode):
        # denoise
        if mode == "train":
            dn_logits, _ = self.generator(nx, labels, x, max_len, self.tau, self.mle)
        else:
            dn_logits, _ = self.generator(nx, labels, None, max_len, self.tau, self.mle)

        # back-translation
        with torch.no_grad():
            _, sample_p = self.generator(x, 1 - labels, None, None, self.tau, self.mle)
            tsf_x = sample_p.argmax(-1)

        if mode == "train":
            bt_logits, _ = self.generator(tsf_x, labels, x, max_len, self.tau, self.mle)
        else:
            bt_logits, _ = self.generator(tsf_x, labels, None, max_len, self.tau, self.mle)

        return dn_logits, bt_logits, tsf_x 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        nx, _, x, _, labels = batch
        dn_logits, bt_logits, _ = self(nx, labels, x, x.size(1), mode="train")
        dn_loss = self.criterion(dn_logits.reshape(-1, dn_logits.size(-1)), x.reshape(-1))
        bt_loss = self.criterion(bt_logits.reshape(-1, bt_logits.size(-1)), x.reshape(-1))
        
        return {
            "loss": dn_loss + bt_loss,
            "progress_bar": {"dn_loss": dn_loss, "bt_loss": bt_loss, "tau": self.tau},
            "log": {"dn_loss": dn_loss, "bt_loss": bt_loss},
        }
    
    def validation_step(self, batch, batch_idx):
        nx, _, x, _, labels = batch
        dn_logits, bt_logits, _ = self(nx, labels, x, x.size(1), mode="inference")
        dn_loss = self.criterion(dn_logits.reshape(-1, dn_logits.size(-1)), x.reshape(-1))
        bt_loss = self.criterion(bt_logits.reshape(-1, bt_logits.size(-1)), x.reshape(-1))

        return {"dn_loss": dn_loss.item(), "bt_loss": bt_loss.item()}
    
    def validation_end(self, outputs):
        dn_losses, bt_losses = [], []
        for output in outputs:
            dn_losses.append(output["dn_loss"])
            bt_losses.append(output["bt_loss"])
        dn_losses, bt_losses = np.array(dn_losses), np.array(bt_losses)

        return {
            "progress_bar": {"dn_loss": dn_losses.mean(), "bt_loss": bt_losses.mean()},
            "log": {"val_loss": (bt_losses.mean().item() - 0.5) ** 2} # let bt loss close to 0.5
        }
    
    def test_step(self, batch, batch_idx):
        nx, _, x, _, labels = batch
        _, _, tsf_x = self(nx, labels, x, x.size(1), mode="inference")

        return {
            "origin_x": x.cpu().numpy().tolist(),
            "tsf_x": tsf_x.cpu().numpy().tolist()
        }

    def test_end(self, outputs):
        for output in outputs:
            triples = zip(output["origin_x"], output["tsf_x"])
            for x, tsf_x in triples:
                print(self.vocab.IdsToSent(x))
                print(self.vocab.IdsToSent(tsf_x, remove_special=False))
                print('-' * 30)
        return {}
    
    @pl.data_loader
    def train_dataloader(self):
        dataset = StyleDataset([f"{self.data_dir}/style.train.0", f"{self.data_dir}/style.train.1"], self.vocab, 
                                max_len=self.args.max_len, load_func=load_s2l)
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, 
                                num_workers=self.args.n_workers, collate_fn=collate_s2s_noise)
        return data_loader
    
    @pl.data_loader
    def val_dataloader(self):
        dataset = StyleDataset([f"{self.data_dir}/style.dev.0", f"{self.data_dir}/style.dev.1"], self.vocab, 
                                max_len=self.args.max_len, load_func=load_s2l)
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, 
                                num_workers=self.args.n_workers, collate_fn=collate_s2s_noise)
        return data_loader
    
    @pl.data_loader
    def test_dataloader(self):
        dataset = StyleDataset([f"{self.data_dir}/style.test.0", f"{self.data_dir}/style.test.1"], self.vocab, 
                                max_len=self.args.max_len, load_func=load_s2l)
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, 
                                num_workers=self.args.n_workers, collate_fn=collate_s2s_noise)
        return data_loader

def construct_trainer(args):
    logger = TensorBoardLogger(save_dir=args.log_dir,
                               name=STAGE,
                               version=args.restore_version)
    checkpoint = ModelCheckpoint(filepath=args.task_dump_dir,
                                 save_weights_only=True,
                                 save_top_k=1,
                                 verbose=0,
                                 monitor='val_loss',
                                 mode='min',
                                 prefix=STAGE)
    early_stop = EarlyStopping(monitor="val_loss",
                               patience=2,
                               mode="min")
    trainer = Trainer(logger=logger,
                      gradient_clip_val=0.5,
                      checkpoint_callback=checkpoint,
                      early_stop_callback=early_stop,
                      max_epochs=args.epochs,
                      gpus=args.device)
    return trainer


if __name__ == "__main__":
    # python main_warmup.py --dataset=[dataset] --mode=train
    from arguments import fetch_args
    args = fetch_args()

    if args.dataset == "yelp":
        args.epochs = 10
        args.batch_size = 200
    elif args.dataset == "gyafc":
        args.epochs = 10
        args.batch_size = 100
    else:
        raise ValueError

    if not os.path.exists(f"{args.dump_dir}/{args.dataset}/{STAGE}"):
        os.mkdir(f"{args.dump_dir}/{args.dataset}/{STAGE}")
    args.task_dump_dir = f"{args.dump_dir}/{args.dataset}/{STAGE}"
    args.log_dir = f"{args.log_dir}/{args.dataset}"

    if args.mode == "train":
        model = CoarseTransfer(args)
        trainer = construct_trainer(args)
        trainer.fit(model)
    elif args.mode == "test":
        import warnings
        warnings.filterwarnings("ignore")

        model = CoarseTransfer(args)
        trainer = construct_trainer(args)
        trainer.test(model)
