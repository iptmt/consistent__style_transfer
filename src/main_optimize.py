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

from model.seq2seq import RNNSearch
from model.classifier import TextCNN
from model.match import Matcher
from model.bilm import BiLM
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
        self.lm = BiLM(len(self.vocab), n_class=2)

        self.generator = RNNSearch(len(self.vocab), args.d_embed, args.d_enc_hidden, args.d_dec_hidden,
                                   args.n_enc_layer, args.n_dec_layer, args.n_class, args.p_drop, args.max_len)
        self.disc = RelGAN_D(len(self.vocab))
        
        # reload pretrained models
        self.classifier.load_state_dict(torch.load(f"{args.dump_dir}/{args.dataset}/pretrain/cls.pth"))
        self.matcher.load_state_dict(torch.load(f"{args.dump_dir}/{args.dataset}/pretrain/mat.pth"))
        self.lm.load_state_dict(torch.load(f"{args.dump_dir}/{args.dataset}/pretrain/lm.pth"))

        self.generator.load_state_dict(torch.load(f"{args.dump_dir}/{args.dataset}/warmup/G.pth"))

        self.data_dir = f"{args.data_dir}/{args.dataset}"

        self.bce_crit = nn.BCEWithLogitsLoss()
        self.ce_crit = nn.CrossEntropyLoss()
        self.mse_crit = nn.MSELoss()
        
        self.tau = args.tau
        self.sigmoid = nn.Sigmoid()

        n_batch = math.ceil(args.n_samples / args.batch_size)
        self.anneal_steps = args.epochs * n_batch
    
    def forward(self, x, labels, tau, optimizer_idx):
        # optimize D
        if optimizer_idx == 1:
            with torch.no_grad():
                _, sample_p = self.generator(x, labels, None, None, gumbel=True, tau=tau)
            t_logits = self.disc(F.one_hot(x, len(self.vocab)).float())
            f_logits = self.disc(sample_p.detach())
            return t_logits, f_logits
        elif optimizer_idx == 0 or optimizer_idx == 2:
            _, sample_p = self.generator(x, labels, None, None, gumbel=True, tau=tau)
            return sample_p
 
    def configure_optimizers(self):
        optimizer_opt = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        optimizer_dsc = torch.optim.Adam(self.disc.parameters(), lr=1e-4)
        optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        return optimizer_opt, optimizer_dsc, optimizer_gen
    
    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        # update discriminator every 1 steps
        if optimizer_idx == 1:
            if batch_idx % 10 == 0:
                optimizer.step()
                optimizer.zero_grad()
        # update generator opt every 1 steps
        if optimizer_idx == 0 or optimizer_idx == 2:
            optimizer.step()
            optimizer.zero_grad()

    def get_current_w(self):
        p = self.global_step / self.anneal_steps
        w = min([p, 1.0])
        tau = self.tau ** p
        return tau, w

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, labels = batch
        tau, w = self.get_current_w()
        # optimize generator with estimators
        if optimizer_idx == 0:
            sample_p = self.forward(x, 1 - labels, self.tau, optimizer_idx)

            s_logits = self.classifier(sample_p)
            c_logits = self.matcher(sample_p, x) 
            l_logits = self.lm(sample_p, 1 - labels)

            s_loss = self.ce_crit(s_logits, 1 - labels)
            c_loss = self.mse_crit(c_logits, c_logits.new_full([c_logits.size(0)], self.hparams.gap))
            l_loss = self.ce_crit(l_logits.reshape(-1, l_logits.size(-1)), sample_p.argmax(-1).reshape(-1))

            loss = w * (self.hparams.alpha * s_loss + self.hparams.beta * c_loss + self.hparams.gamma * l_loss)
            loginfo = {"s": s_loss, "c": c_loss, "l": l_loss}
            return {"loss": loss, "progress_bar": loginfo, "log": loginfo}

        # optimize discriminator
        if optimizer_idx == 1:
            t_logits, f_logits = self.forward(x, 1 - labels, tau, optimizer_idx)
            t_labels, f_labels = t_logits.new_ones([t_logits.size(0)]), f_logits.new_zeros([f_logits.size(0)])
            d_loss = 0.5 * (self.bce_crit(t_logits, t_labels) + self.bce_crit(f_logits, f_labels))
            loginfo = {"D": d_loss}
            return {"loss": d_loss, "progress_bar": loginfo, "log": loginfo}
        
        # optimize generator
        if optimizer_idx == 2:
            sample_p = self.forward(x, 1 - labels, tau, optimizer_idx)
            g_logits = self.disc(sample_p)
            g_labels = g_logits.new_ones([g_logits.size(0)])
            g_loss = self.bce_crit(g_logits, g_labels)

            loginfo = {"G": g_loss}
            return {"loss": g_loss, "progress_bar": loginfo, "log": loginfo}
        
    def validation_step(self, batch, batch_idx):
        x, labels = batch

        _, sample_p = self.generator(x, 1 - labels, None, None, gumbel=True, tau=self.tau)

        s_logits = self.classifier(sample_p)
        c_logits = self.matcher(sample_p, x) 
        l_logits = self.lm(sample_p, 1 - labels)

        s_loss = self.ce_crit(s_logits, 1 - labels)
        c_loss = c_logits.mean()
        l_loss = self.ce_crit(l_logits.reshape(-1, l_logits.size(-1)), sample_p.argmax(-1).reshape(-1))

        return {"loss": s_loss.item() + c_loss.item() + l_loss.item()}
        
 
    def validation_end(self, outputs):
        val_loss = sum([output["loss"] for output in outputs]) / len(outputs)
        return {
            "progress_bar": {"val_loss": val_loss},
            "log": {"val_loss": val_loss}
        }
    
    def test_step(self, batch, batch_idx):
        x, labels = batch
        logits, _ = self.generator(x, labels, None, None, gumbel=False)
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
                                 save_weights_only=False,
                                 save_top_k=1,
                                 verbose=0,
                                 monitor='val_loss',
                                 mode='min',
                                 prefix=STAGE)
    early_stop = EarlyStopping(monitor="val_loss",
                               patience=2,
                               mode="min")
    trainer = Trainer(logger=logger,
                      gradient_clip_val=1.0,
                      checkpoint_callback=checkpoint,
                      early_stop_callback=early_stop,
                      max_epochs=args.epochs,
                      gpus=args.device)
    return trainer        


if __name__ == "__main__":
    # python main_optimize.py --dataset=[dataset] --mode=train
    from arguments import fetch_args
    args = fetch_args()

    if args.dataset == "yelp":
        args.epochs = 20
        args.batch_size = 200
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
