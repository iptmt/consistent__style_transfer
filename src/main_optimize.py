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
        self.lm = BiLM(len(self.vocab))

        self.generator = DenoiseTransformer(len(self.vocab), args.n_class, args.max_len)
        
        # reload pretrained models
        self.classifier.load_state_dict(torch.load(f"{args.dump_dir}/{args.dataset}/pretrain/cls.pth"))
        self.matcher.load_state_dict(torch.load(f"{args.dump_dir}/{args.dataset}/pretrain/mat.pth"))
        self.lm.load_state_dict(torch.load(f"{args.dump_dir}/{args.dataset}/pretrain/lm.pth"))

        self.generator.load_state_dict(torch.load(f"{args.dump_dir}/{args.dataset}/warmup/G.pth"))

        self.data_dir = f"{args.data_dir}/{args.dataset}"

        self.ce_crit = nn.CrossEntropyLoss()
        self.mse_crit = nn.MSELoss()
        
        self.tau = args.tau
        self.softmax = nn.Softmax(-1)

        n_batch = math.ceil(args.n_samples / args.batch_size)
        self.anneal_steps = args.epochs * n_batch
    
    def forward(self, x, labels, tau):
        # sample_p = self.generator(x, labels, None, gumbel=True, tau=tau)
        sample_p = self.generator(x, labels, None)
        sample_p = self.softmax(sample_p / tau)
        return sample_p
 
    def configure_optimizers(self):
        optimizer_opt = torch.optim.Adam(self.generator.parameters(), lr=1e-5)
        return optimizer_opt
    
    def soft_ce(self, s, t):
        return - (t * F.log_softmax(s, -1)).sum(-1).mean()

    def training_step(self, batch, batch_idx):
        x, labels = batch
        sample_p = self.forward(x, 1 - labels, self.tau)

        s_logits = self.classifier(sample_p)
        c_logits = self.matcher(sample_p, x) 
        l_logits = self.lm(sample_p, 1 - labels)

        s_loss = self.ce_crit(s_logits, 1 - labels)
        c_loss = self.mse_crit(c_logits, c_logits.new_full([c_logits.size(0)], self.hparams.gap))
        l_loss = self.soft_ce(l_logits, sample_p)

        loss = self.hparams.w_s * s_loss + self.hparams.w_c * c_loss + self.hparams.w_l * l_loss
        loginfo = {"S": s_loss, "C": c_loss, "L": l_loss}
        return {"loss": loss, "progress_bar": loginfo, "log": loginfo}

    def validation_step(self, batch, batch_idx):
        x, labels = batch

        # sample_p = self.generator(x, 1 - labels, None, gumbel=True, tau=self.tau)
        sample_p = self.forward(x, 1 - labels, self.tau)

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
        args.batch_size = 256
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
