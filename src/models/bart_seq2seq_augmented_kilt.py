import math
from argparse import ArgumentParser
from copy import deepcopy

import pytorch_lightning as pl
import torch
from higher.patch import monkeypatch as make_functional
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    get_constant_schedule,
    get_linear_schedule_with_warmup,
)

from src.data.seq2seq_augmented_kilt import Seq2SeqAugmentedKILT
from src.models.bart_seq2seq_kilt import BartSeq2Seq
from src.models.one_shot_learner import OneShotLearner
from src.utils import batch_it, label_smoothed_nll_loss


class BartSeq2SeqAugmented(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--train_data_path",
            type=str,
            default="../datasets/structured_zeroshot-train-new_annotated_final.jsonl",
        )
        parser.add_argument(
            "--dev_data_path",
            type=str,
            default="../kilt/datasets/structured_zeroshot-dev-new_annotated_final.jsonl",
        )
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--lr_alpha", type=float, default=1e-1)
        parser.add_argument("--max_length", type=int, default=32)
        parser.add_argument("--weight_decay", type=int, default=0.01)
        parser.add_argument("--total_num_updates", type=int, default=200000)
        parser.add_argument("--warmup_updates", type=int, default=1000)
        parser.add_argument("--num_workers", type=int, default=0)

        parser.add_argument("--model_name", type=str, default="facebook/bart-base")
        parser.add_argument("--eps", type=float, default=0.1)
        parser.add_argument(
            "--model_checkpoint",
            type=str,
            default="models/QA_model.ckpt",
        )

        parser.add_argument("--margin_kl_max", type=float, default=1e-3)
        parser.add_argument("--margin_kl_min", type=float, default=1e-5)
        parser.add_argument("--margin_lp_max", type=float, default=1e-3)
        parser.add_argument("--margin_lp_min", type=float, default=1e-7)
        parser.add_argument("--max_scale", type=float, default=1)
        parser.add_argument("--p", type=float, default=2)
        parser.add_argument(
            "--divergences", type=str, choices=["kl", "lp", "both"], default="kl"
        )
        parser.add_argument("--use_views", action="store_true")

        return parser

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = BartTokenizer.from_pretrained(self.hparams.model_name)
        self.model = BartSeq2Seq.load_from_checkpoint(
            self.hparams.model_checkpoint
        ).model.eval()

        self.learner = OneShotLearner(
            self.model,
            vocab_dim=self.model.model.shared.weight.data.shape[0],
            embedding_dim=self.model.model.shared.weight.data.shape[1],
            hidden_dim=128,
            condition_dim=1024,
            include_set={
                n
                for n, _ in self.model.named_parameters()
                if all(
                    e not in n.lower()
                    for e in (
                        "bias",
                        "norm",
                        "embeddings",
                        "classifier",
                        "pooler",
                        "shared",
                        "embed",
                        "positions",
                    )
                )
            },
            max_scale=self.hparams.max_scale,
            embedding_init=self.model.model.shared.weight.data,
        )

        self.alpha_kl = torch.nn.Parameter(torch.ones(()))
        self.alpha_kl.register_hook(lambda grad: -grad)

        self.alpha_lp = torch.nn.Parameter(torch.ones(()))
        self.alpha_lp.register_hook(lambda grad: -grad)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.valid_flipped = pl.metrics.Accuracy()

        self.register_buffer("margin_kl", torch.tensor(self.hparams.margin_kl_max))
        self.register_buffer("margin_lp", torch.tensor(self.hparams.margin_lp_max))
        self.running_flipped = []

    def train_dataloader(self, shuffle=True):
        if not hasattr(self, "train_dataset"):
            self.train_dataset = Seq2SeqAugmentedKILT(
                tokenizer=self.tokenizer,
                data_path=self.hparams.train_data_path,
                max_length=self.hparams.max_length,
                return_view=self.hparams.use_views,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )

    def val_dataloader(self, shuffle=True):
        if not hasattr(self, "val_dataset"):
            self.val_dataset = Seq2SeqAugmentedKILT(
                tokenizer=self.tokenizer,
                data_path=self.hparams.dev_data_path,
                max_length=self.hparams.max_length,
                return_view=self.hparams.use_views,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )

    def get_logits_orig_params_dict(self, batch):

        with torch.enable_grad():
            logits_orig, logit_for_grad, _ = self.model.eval()(
                input_ids=batch["src_input_ids"],
                attention_mask=batch["src_attention_mask"],
                decoder_input_ids=batch["trg_input_ids"][:, :-1],
                decoder_attention_mask=batch["trg_attention_mask"][:, :-1],
                use_cache=False,
            ).logits.split(
                [
                    len(batch["src_input_ids"]) - (2 if self.hparams.use_views else 1),
                    1,
                    1 if self.hparams.use_views else 0,
                ]
            )

            logits_orig = logits_orig.detach()

            grads = torch.autograd.grad(
                label_smoothed_nll_loss(
                    logit_for_grad.log_softmax(-1),
                    batch["trg_input_ids"][
                        -2
                        if self.hparams.use_views
                        else -1 : -1
                        if self.hparams.use_views
                        else None, 1:
                    ],
                    epsilon=self.hparams.eps,
                    ignore_index=self.tokenizer.pad_token_id,
                )[1]
                / batch["trg_attention_mask"][
                    -2
                    if self.hparams.use_views
                    else -1 : -1
                    if self.hparams.use_views
                    else None, 1:
                ].sum(),
                self.model.parameters(),
            )
            grads = {
                name: grad
                for (name, _), grad in zip(self.model.named_parameters(), grads)
            }

        params_dict = self.learner(
            batch["cond_input_ids"],
            batch["cond_attention_mask"],
            grads=grads,
        )

        return logits_orig, params_dict

    def forward(self, batch):

        logits_orig, params_dict = self.get_logits_orig_params_dict(batch)
        fmodel = make_functional(self.model).eval()

        logits = fmodel(
            input_ids=batch["src_input_ids"],
            attention_mask=batch["src_attention_mask"],
            decoder_input_ids=batch["trg_input_ids"][:, :-1],
            decoder_attention_mask=batch["trg_attention_mask"][:, :-1],
            use_cache=False,
            params=[
                params_dict.get(n, 0) + p for n, p in self.model.named_parameters()
            ],
        ).logits

        return logits_orig, logits, params_dict

    def get_kl_lp_cr(self, logits_orig, logits, labels, params_dict):

        kl = torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=logits_orig),
            torch.distributions.Categorical(
                logits=logits[: -2 if self.hparams.use_views else -1]
            ),
        )

        lp = sum(
            (p.abs() ** self.hparams.p).mean() ** (1 / self.hparams.p)
            for p in params_dict.values()
        ) / len(params_dict)

        cr, _ = label_smoothed_nll_loss(
            logits[-2 if self.hparams.use_views else -1 :].log_softmax(-1),
            labels[-2 if self.hparams.use_views else -1 :],
            epsilon=self.hparams.eps,
            ignore_index=self.tokenizer.pad_token_id,
        )

        return kl, lp, cr

    def training_step(self, batch, batch_idx):

        logits_orig, logits, params_dict = self.forward(batch)

        kl, lp, cr = self.get_kl_lp_cr(
            logits_orig, logits, batch["trg_input_ids"][:, 1:], params_dict
        )

        kl = (
            kl
            * batch["trg_attention_mask"][: (-2 if self.hparams.use_views else -1), 1:]
        ).sum() / batch["trg_attention_mask"][
            : (-2 if self.hparams.use_views else -1), 1:
        ].sum()

        cr = (
            cr
            / batch["trg_attention_mask"][
                (-2 if self.hparams.use_views else -1) :, 1:
            ].sum()
        )

        loss_kl = self.alpha_kl * (kl - self.margin_kl)
        loss_lp = self.alpha_lp * (lp - self.margin_lp)

        if self.hparams.divergences == "both":
            loss = cr + loss_kl + loss_lp
        elif self.hparams.divergences == "kl":
            loss = cr + loss_kl
        elif self.hparams.divergences == "lp":
            loss = cr + loss_lp

        self.log("alpha_kl", self.alpha_kl, on_step=True, on_epoch=False, prog_bar=True)
        self.log("alpha_lp", self.alpha_lp, on_step=True, on_epoch=False, prog_bar=True)
        self.log("kl", kl, on_step=True, on_epoch=False, prog_bar=True)
        self.log("lp", lp, on_step=True, on_epoch=False, prog_bar=True)
        self.log("cr", cr, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx=None):

        _, params_dict = self.get_logits_orig_params_dict(batch)

        fmodel = make_functional(self.model).eval()

        gold = [b["pred"] for b in batch["raw"][:-1]] + [batch["raw"][-1]["alt"]] * (
            2 if self.hparams.use_views else 1
        )

        guess = self.tokenizer.batch_decode(
            fmodel.generate(
                input_ids=batch["src_input_ids"],
                attention_mask=batch["src_attention_mask"],
                min_length=0,
                num_beams=5,
                num_return_sequences=1,
                params=[
                    params_dict.get(n, 0) + p for n, p in self.model.named_parameters()
                ],
            ),
            skip_special_tokens=True,
        )

        acc = torch.tensor(
            [a.lower().strip() == b.lower().strip() for a, b in zip(guess, gold)]
        ).long()
        self.valid_acc(acc, torch.ones_like(acc))
        self.log(
            "valid_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        self.valid_flipped(
            acc[(-2 if self.hparams.use_views else -1) :],
            torch.ones_like(acc[(-2 if self.hparams.use_views else -1) :]),
        )
        self.log(
            "valid_flipped",
            self.valid_flipped,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def sample(self, sentences, condition, params_dict=None, num_return_sequences=1, stop_condition=None):
        len_sent = len(sentences)
        with torch.no_grad():
            batch = {
                k: v.to(self.device)
                for k, v in self.val_dataset.get_batch(sentences, condition).items()
            }

            if not params_dict:
                _, params_dict = self.get_logits_orig_params_dict(batch)

            fmodel = make_functional(self.model).eval()

            guess = list(
                batch_it(
                    self.tokenizer.batch_decode(
                        fmodel.generate(
                            input_ids=batch["src_input_ids"],
                            attention_mask=batch["src_attention_mask"],
                            min_length=0,
                            num_beams=5,
                            num_return_sequences=num_return_sequences,
                            params=[
                                params_dict.get(n, 0) + p
                                for n, p in self.model.named_parameters()
                            ],
                        ),
                        skip_special_tokens=True,
                    ),
                    num_return_sequences,
                )
            )
            
            n_iter = 1
            if stop_condition is not None and stop_condition(condition, guess, n_iter):
                model_tmp = deepcopy(self.model)
                params_dict_tmp =  deepcopy(params_dict)
                
                while stop_condition(condition, guess, n_iter):
                    for n, p in self.model.named_parameters():
                        p.data += params_dict.get(n, 0)
                        
                    guess = list(
                        batch_it(
                            self.tokenizer.batch_decode(
                                fmodel.generate(
                                    input_ids=batch["src_input_ids"],
                                    attention_mask=batch["src_attention_mask"],
                                    min_length=0,
                                    num_beams=5,
                                    num_return_sequences=num_return_sequences,
                                    params=[
                                        params_dict.get(n, 0) + p
                                        for n, p in self.model.named_parameters()
                                    ],
                                ),
                                skip_special_tokens=True,
                            ),
                            num_return_sequences,
                        )
                    )
                        
                    params_dict_tmp = {k: v + params_dict[k] for k, v in params_dict_tmp.items()}
                    n_iter += 1
                
                self.model = model_tmp
                params_dict = params_dict_tmp

            if num_return_sequences == 1:
                guess = [e[0] for e in guess]

            return params_dict, guess[:len_sent]

    def on_before_zero_grad(self, optimizer):
        self.alpha_kl.data = torch.where(
            self.alpha_kl.data < 0,
            torch.full_like(self.alpha_kl.data, 0),
            self.alpha_kl.data,
        )
        self.alpha_lp.data = torch.where(
            self.alpha_lp.data < 0,
            torch.full_like(self.alpha_lp.data, 0),
            self.alpha_lp.data,
        )

    def on_validation_epoch_end(self):
        if self.valid_flipped.compute().item() > 0.9:
            self.margin_kl = max(
                self.margin_kl * 0.8, self.margin_kl * 0 + self.hparams.margin_kl_min
            )
            self.margin_lp = max(
                self.margin_lp * 0.8, self.margin_lp * 0 + self.hparams.margin_lp_min
            )
        self.log(
            "margin_kl", self.margin_kl, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "margin_lp", self.margin_lp, on_step=False, on_epoch=True, prog_bar=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            [
                {
                    "params": self.learner.parameters(),
                    "lr": self.hparams.lr,
                },
                {
                    "params": [self.alpha_kl, self.alpha_lp],
                    "lr": self.hparams.lr_alpha,
                },
            ],
            centered=True,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_updates,
            num_training_steps=self.hparams.total_num_updates,
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]
