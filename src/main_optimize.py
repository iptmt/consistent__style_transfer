import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model.albert import AlbertTeacher
from model.generator import RNNSearch
from model.classifier import TextCNN

from vocab import Wrapped_ALBERT_Vocab
from loader import StyleDataset, load_s2l, collate_s2s_noise 
from data_util import from_output_to_input
from main_warmup import CoarseTransfer
from main_albert_ft import AlbertTuner

from transformers import AlbertConfig

STAGE = "optimize"

class GenerationTuner(pl.LightningModule):
    def __init__(self, args):
        super(GenerationTuner, self).__init__()

        self.args = args
        self.hparams = args

        self.vocab = Wrapped_ALBERT_Vocab(args.dump_dir)

        # construct new models
        self.generator = RNNSearch(args.n_vocab, args.d_embed, args.d_enc_hidden, args.d_dec_hidden, 
                                   args.n_enc_layer, args.n_dec_layer, args.n_class, args.p_drop, args.max_len)
        self.albert = AlbertTeacher(AlbertConfig.from_pretrained(args.dump_dir))
        self.classifier = TextCNN(args.n_vocab)
        
        # reload pretrained models
        pretrained_generator = CoarseTransfer.load_from_checkpoint(args.gen_dump_dir)
        pretrained_albert = AlbertTuner.load_from_checkpoint(args.albert_dump_dir)

        # weight initialization
        self.generator.load_state_dict(pretrained_generator.generator.state_dict())
        self.albert.load_state_dict(pretrained_albert.albert.state_dict())
        self.classifier.load_state_dict(pretrained_albert.classifier.state_dict())

        self.data_dir = f"{args.data_dir}/{args.dataset}"

        self.ce_crit = nn.CrossEntropyLoss(ignore_index=-1)
        self.softmax = nn.Softmax(dim=-1)
        
        self.tau = args.tau
        self.mle = args.mle
        self.hard = args.hard

        n_batch = math.ceil(self.args.n_samples / self.args.batch_size)
        self.anneal_steps = self.args.epochs * n_batch
    
    def forward(self, nx, labels, x, len_x, max_len, tau):
        # denoise
        dn_logits, _ = self.generator(nx, labels, x, max_len, tau, self.mle, self.hard)

        # transfer
        _, sample_p = self.generator(x, 1 - labels, None, None, tau, self.mle, self.hard)

        padded_scaled_logits, trunc_len, complement_input, input_mask, attn_mask, segments, complement_output, output_mask = \
            from_output_to_input(sample_p, x, len_x, len(self.vocab))

        sample_p = sample_p[:, :trunc_len, :]
        tsf_tokens = sample_p.argmax(-1)

        # optimize
        input_mask = input_mask.unsqueeze(-1)
        inputs = input_mask * complement_input + (1 - input_mask) * padded_scaled_logits

        c_logits, mlm_logits = self.albert(inputs, attention_mask=attn_mask, token_type_ids=segments, labels=1-labels)

        output_mask = output_mask.unsqueeze(-1)
        mlm_labels = complement_output * output_mask + sample_p * (1 - output_mask)
        s_logits = self.classifier(mlm_labels)

        mlm_logits = complement_output * output_mask + mlm_logits[:, 1: 1 + trunc_len, :] * (1 - output_mask)

        # back translate
        bt_logits, _ = self.generator(tsf_tokens, labels, x, max_len, tau, self.mle, self.hard)

        return (
            (dn_logits, x), (bt_logits, x), (s_logits, 1 - labels), (c_logits, None), (mlm_logits, mlm_labels, 1 - output_mask)
        )
 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr)
        scheduler = LambdaLR(optimizer, lambda epoch: 1.0 ** epoch)
        return [optimizer], [scheduler]

    # s -> student; P_t -> teacher distribution
    def cal_soft_ce_loss(self, s, P_t, mask):
        return -(P_t * F.log_softmax(s, dim=-1) * mask).sum(-1).mean()
    
    def get_current_w(self):
        p = self.global_step / self.anneal_steps
        w = min([p, 1.0])
        tau = self.tau ** p
        return self.tau, w

    def training_step(self, batch, batch_idx):
        
        tau, w = self.get_current_w()

        nx, _, x, len_x, labels = batch

        ((dn_logits, lb_dn), (bt_logits, lb_bt), (s_logits, lb_s), (c_logits, _), (mlm_logits, lb_lm, mask)) = self(nx, labels, x, len_x, x.size(1), tau)
        
        dn_loss = self.ce_crit(dn_logits.reshape(-1, dn_logits.size(-1)), lb_dn.reshape(-1))
        bt_loss = self.ce_crit(bt_logits.reshape(-1, bt_logits.size(-1)), lb_bt.reshape(-1))
        s_loss = self.ce_crit(s_logits, lb_s)
        c_loss = ((1 - c_logits) ** 2).mean()
        mlm_loss = self.cal_soft_ce_loss(mlm_logits, lb_lm, mask)

        w_c, w_mlm, w_s = self.args.alpha, self.args.beta, self.args.gamma

        return {
            'loss': bt_loss + dn_loss + w * (w_c * c_loss + w_mlm * mlm_loss + w_s * s_loss),
            'progress_bar': {"s": s_loss, "c": (1 - c_logits).mean(), "mlm": mlm_loss, "tau": tau},
            'log': {'L_dn': dn_loss, 'L_bt': bt_loss, "L_s": s_loss, "L_c": c_loss, "L_mlm": mlm_loss, "w": w, "tau": tau}
        }
    
    def validation_step(self, batch, batch_idx):
        nx, _, x, len_x, labels = batch
        ((_, _), (_, _), (s_logits, lb_s), (c_logits, _), (mlm_logits, lb_lm, mask)) = self(nx, labels, x, len_x, x.size(1), self.tau)
        
        s_loss = self.ce_crit(s_logits, lb_s)
        c_loss = (1 - c_logits).mean()
        mlm_loss = self.cal_soft_ce_loss(mlm_logits, lb_lm, mask)

        return {
            "L_s": s_loss.item(), "L_c": c_loss.item(), "L_mlm": mlm_loss.item()
        }
 
    def validation_end(self, outputs):
        val_loss = sum([output['L_s'] + output["L_c"] + output["L_mlm"] for output in outputs]) / len(outputs)
        return {
            "progress_bar": {"val_loss": val_loss},
            "log": {"val_loss": val_loss}
        }
    
    def test_step(self, batch, batch_idx):
        _, _, x, _, labels = batch
        _, sample_p = self.generator(x, 1 - labels, None, None, mle=True)
        return {
            "ori": x.cpu().numpy().tolist(),
            "tsf": sample_p.argmax(-1).cpu().numpy().tolist(),
            "label": labels.cpu().numpy().tolist()
        }
    
    def test_end(self, outputs):
        print(f"Writing outputs to {self.args.out_dir}/")
        with open(f"{self.args.out_dir}/style.{args.test_file}.0.tsf", 'w+', encoding='utf-8') as f_0:
            with open(f"{self.args.out_dir}/style.{args.test_file}.1.tsf", 'w+', encoding='utf-8') as f_1:
                for output in outputs:
                    for _, tsf, label in zip(output["ori"], output["tsf"], output["label"]):
                        f = f_0 if label == 0 else f_1
                        f.write(self.vocab.IdsToSent(tsf) + "\n")
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
        dataset = StyleDataset([f"{self.data_dir}/style.{args.test_file}.0", f"{self.data_dir}/style.{args.test_file}.1"], self.vocab, 
                                max_len=self.args.max_len, load_func=load_s2l)
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, 
                                num_workers=self.args.n_workers, collate_fn=collate_s2s_noise)
        return data_loader

def construct_trainer(args):
    logger = TensorBoardLogger(save_dir=args.log_dir,
                               name=f"{STAGE}-{args.model_version}",
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
                      gradient_clip_val=0.5,
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
        args.epochs = 10
        args.batch_size = 100
    elif args.dataset == "gyafc":
        args.epochs = 10
        args.batch_size = 50
    else:
        raise ValueError

    # load generator
    dirs = os.listdir(f"{args.dump_dir}/{args.dataset}/warmup")
    dirs.sort()
    args.gen_dump_dir = f"{args.dump_dir}/{args.dataset}/warmup/{dirs[-1]}"

    # load albert
    dirs = os.listdir(f"{args.dump_dir}/{args.dataset}/albert")
    dirs.sort()
    args.albert_dump_dir = f"{args.dump_dir}/{args.dataset}/albert/{dirs[-1]}"

    # dump dir
    if not os.path.exists(f"{args.dump_dir}/{args.dataset}/{STAGE}-{args.model_version}"):
        os.mkdir(f"{args.dump_dir}/{args.dataset}/{STAGE}-{args.model_version}")
    args.task_dump_dir = f"{args.dump_dir}/{args.dataset}/{STAGE}-{args.model_version}"

    # output dir
    if not os.path.exists(f"{args.out_dir}/{args.dataset}-{args.model_version}"):
        os.mkdir(f"{args.out_dir}/{args.dataset}-{args.model_version}")
    args.out_dir = f"{args.out_dir}/{args.dataset}-{args.model_version}"

    args.log_dir = f"{args.log_dir}/{args.dataset}"

    if args.mode == "train":
        args.test_file = "test"
        model = GenerationTuner(args)
        trainer = construct_trainer(args)
        trainer.fit(model)
    elif args.mode == "test":
        import warnings
        warnings.filterwarnings("ignore")

        for test_file in ("train", "test"):
            # special parameter
            args.test_file = test_file
            # special parameter
            model = GenerationTuner(args)

            dirs = os.listdir(args.task_dump_dir)
            if len(dirs) > 0:
                dirs.sort()
                args.tsf_dump_dir = f"{args.task_dump_dir}/{dirs[-1]}"
                pretrain_model = GenerationTuner.load_from_checkpoint(args.tsf_dump_dir)
                model.load_state_dict(pretrain_model.state_dict())
            model.args = args
            trainer = construct_trainer(args)
            trainer.test(model)
