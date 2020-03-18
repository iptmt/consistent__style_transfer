import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model.transformer import DenoiseTransformer
from model.classifier import TextCNN
from model.match import Matcher
from model.mlm import MLM
from model.discriminator import RelGAN_D

from vocab import BPETokenizer
from loader import StyleDataset, load_s2l, collate_optimize 

STAGE = "optimize"

class GenerationTuner(pl.LightningModule):
    def __init__(self, args):
        super(GenerationTuner, self).__init__()

        self.hparams = args

        self.vocab = BPETokenizer.load(f"{args.dump_dir}/{args.dataset}/{args.dataset}-vocab.json",
                                       f"{args.dump_dir}/{args.dataset}/{args.dataset}-merges.txt")
        # construct new models
        self.classifier = TextCNN(len(self.vocab), n_class=2)
        self.matcher = Matcher(len(self.vocab))
        # self.lm = MLM(len(self.vocab), args.n_class)

        self.disc = RelGAN_D(len(self.vocab))

        self.generator = DenoiseTransformer(len(self.vocab), args.n_class, args.max_len)
        
        # reload pretrained models
        self.classifier.load_state_dict(torch.load(f"{args.dump_dir}/{args.dataset}/pretrain/cls.pth"))
        self.matcher.load_state_dict(torch.load(f"{args.dump_dir}/{args.dataset}/pretrain/mat.pth"))
        # self.lm.load_state_dict(torch.load(f"{args.dump_dir}/{args.dataset}/pretrain/lm.pth"))

        self.generator.load_state_dict(torch.load(f"{args.dump_dir}/{args.dataset}/warmup/G.pth"))

        self.data_dir = f"{args.data_dir}/{args.dataset}"

        self.ce_crit = nn.CrossEntropyLoss()
        self.mse_crit = nn.MSELoss()
        self.bce_crit = nn.BCEWithLogitsLoss()
        
        self.tau = args.tau
        self.softmax = nn.Softmax(-1)

        self.ws, self.wc = args.w_s, args.w_c
    
    def forward(self, x, labels, tau):
        sample_p = self.generator(x, labels, None)
        sample_p = self.softmax(sample_p / tau)
        return sample_p
 
    def configure_optimizers(self):
        optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=1e-5)
        optimizer_adv = torch.optim.Adam(self.disc.parameters(), lr=1e-5)
        return optimizer_gen, optimizer_adv
    
    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx, 
                             second_order_closure=None):
        if optimizer_idx == 0:
            optimizer.step()
            optimizer.zero_grad()

        # update discriminator opt every 5 steps
        if optimizer_idx == 1:
            if batch_idx % 5 == 0 :
                optimizer.step()
                optimizer.zero_grad()
    
    def adv_label(self, logits, value):
        return logits.new_full(logits.shape, value)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        nx, x, labels = batch

        tau = self.tau ** min([1.0, self.global_step / 17000])

        if optimizer_idx == 0:
            sample_p = self.forward(nx, 1 - labels, tau)

            s_logits = self.classifier(sample_p)
            c_logits = self.matcher(sample_p, x) 

            self.disc.eval()
            adv_logits = self.disc(sample_p)

            s_loss = self.ce_crit(s_logits, 1 - labels)
            c_loss = self.mse_crit(c_logits, c_logits.new_full([c_logits.size(0)], self.hparams.gap))
            G_loss = self.bce_crit(adv_logits, self.adv_label(adv_logits, 1))

            loss = G_loss + self.wc * c_loss + self.ws * s_loss
            loginfo = {"G": G_loss, "S": s_loss, "C": c_loss}
            return {"loss": loss, "progress_bar": loginfo, "log": loginfo}
        
        if optimizer_idx == 1:
            self.disc.train()
            t_logits = self.disc(F.one_hot(x, len(self.vocab)).float())
            with torch.no_grad():
                x_ = self.forward(x, 1 - labels, tau)
            f_logits = self.disc(x_)

            D_loss = 0.5 * (self.bce_crit(t_logits, self.adv_label(t_logits, 1)) + \
                self.bce_crit(f_logits, self.adv_label(f_logits, 0)))
            return {"loss": D_loss, "progress_bar": {"D": D_loss}, "log": {"D": D_loss}}


    def validation_step(self, batch, batch_idx):
        _, x, labels = batch

        sample_p = self.forward(x, 1 - labels, self.tau)

        s_logits = self.classifier(sample_p)
        c_logits = self.matcher(sample_p, x) 
        adv_logits = self.disc(sample_p)

        s_loss = self.ce_crit(s_logits, 1 - labels)
        c_loss = self.mse_crit(c_logits, c_logits.new_full([c_logits.size(0)], self.hparams.gap))
        G_loss = self.bce_crit(adv_logits, self.adv_label(adv_logits, 1))

        return {"loss": (G_loss + self.ws * s_loss + self.wc * c_loss).item()}
        
 
    def validation_end(self, outputs):
        val_loss = sum([output["loss"] for output in outputs]) / len(outputs)
        return {
            "progress_bar": {"val_loss": val_loss},
            "log": {"val_loss": val_loss}
        }
    
    def test_step(self, batch, batch_idx):
        _, x, labels = batch
        logits = self.forward(x, 1 - labels, self.tau)
        return {
            "ori": x.cpu().numpy().tolist(),
            "tsf": logits.argmax(-1).cpu().numpy().tolist(),
            "label": labels.cpu().numpy().tolist()
        }
    
    def test_end(self, outputs):
        print(f"Writing outputs to {self.hparams.out_dir}/")
        with open(f"{self.hparams.out_dir}/style.{self.hparams.test_file}.0.tsf", 'w+', encoding='utf-8') as f_0:
            with open(f"{self.hparams.out_dir}/style.{self.hparams.test_file}.1.tsf", 'w+', encoding='utf-8') as f_1:
                for output in outputs:
                    for _, tsf, label in zip(output["ori"], output["tsf"], output["label"]):
                        f = f_0 if label == 0 else f_1
                        f.write(self.vocab.decode(tsf) + "\n")
        return {}

    @pl.data_loader
    def train_dataloader(self):
        dataset = StyleDataset([f"{self.data_dir}/style.train.0", f"{self.data_dir}/style.train.1"], self.vocab,
                                max_len=self.hparams.max_len, load_func=load_s2l)
        data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, 
                                 collate_fn=collate_optimize)
        return data_loader
    
    @pl.data_loader
    def val_dataloader(self):
        dataset = StyleDataset([f"{self.data_dir}/style.dev.0", f"{self.data_dir}/style.dev.1"], self.vocab, 
                                max_len=self.hparams.max_len, load_func=load_s2l)
        data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False,
                                 collate_fn=collate_optimize)
        return data_loader
    
    @pl.data_loader
    def test_dataloader(self):
        dataset = StyleDataset([f"{self.data_dir}/style.{self.hparams.test_file}.0", 
                                f"{self.data_dir}/style.{self.hparams.test_file}.1"], self.vocab, 
                                max_len=self.hparams.max_len, load_func=load_s2l)
        data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, 
                                 collate_fn=collate_optimize)
        return data_loader

def construct_trainer(args):
    logger = TensorBoardLogger(save_dir=args.log_dir,
                               name=f"{STAGE}-{args.ver}",
                               version=args.restore_version)
    checkpoint = ModelCheckpoint(filepath=args.task_dump_dir,
                                 save_weights_only=True,
                                 save_top_k=5,
                                 verbose=0,
                                 monitor='val_loss',
                                 mode='min',
                                 prefix=STAGE)
    early_stop = EarlyStopping(monitor="val_loss",
                               patience=5,
                               mode="min")
    trainer = Trainer(logger=logger,
                      early_stop_callback=early_stop,
                      gradient_clip_val=1.0,
                      checkpoint_callback=checkpoint,
                      max_epochs=args.epochs,
                      gpus=args.device)
    return trainer        


if __name__ == "__main__":
    from arguments import fetch_args
    args = fetch_args()

    if args.dataset == "yelp":
        args.epochs = 20
        args.batch_size = 256
    elif args.dataset == "shen":
        args.epochs = 10
        args.batch_size = 256
    elif args.dataset == "book":
        args.batch_size = 128
        args.epochs = 10
    else:
        raise ValueError

    # dump dir
    if not os.path.exists(f"{args.dump_dir}/{args.dataset}/{STAGE}-{args.ver}"):
        os.mkdir(f"{args.dump_dir}/{args.dataset}/{STAGE}-{args.ver}")
    args.task_dump_dir = f"{args.dump_dir}/{args.dataset}/{STAGE}-{args.ver}"

    # output dir
    if not os.path.exists(f"{args.out_dir}/{args.dataset}-{args.ver}"):
        os.mkdir(f"{args.out_dir}/{args.dataset}-{args.ver}")
    args.out_dir = f"{args.out_dir}/{args.dataset}-{args.ver}"

    args.log_dir = f"{args.log_dir}/{args.dataset}"

    if args.mode == "train":
        args.test_file = "test"
        model = GenerationTuner(args)
        trainer = construct_trainer(args)
        trainer.fit(model)
    elif args.mode == "test":
        import warnings
        warnings.filterwarnings("ignore")

        for transfer_file in ("train", "test"):
            # special parameter
            args.test_file = transfer_file
            # special parameter
            model = GenerationTuner(args)

            dirs = os.listdir(args.task_dump_dir)
            if len(dirs) > 0:
                dirs.sort()
                args.tsf_dump_dir = f"{args.task_dump_dir}/{dirs[-1]}"
                pretrain_model = GenerationTuner.load_from_checkpoint(args.tsf_dump_dir)
                model.load_state_dict(pretrain_model.state_dict())
            trainer = construct_trainer(args)
            trainer.test(model)
