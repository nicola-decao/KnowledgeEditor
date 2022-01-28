from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup

from src.data.binary_kilt import BinaryKILT


class BertClassifier(torch.nn.Module):
    def __init__(self, model_name, hidden_dim=768):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def forward(self, *args, **kwargs):
        return self.classifier(self.model(*args, **kwargs)[1]).squeeze(-1)


class BertBinary(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--train_data_path",
            type=str,
            default="../datasets/fever-train-kilt.jsonl",
        )
        parser.add_argument(
            "--dev_data_path",
            type=str,
            default="../datasets/fever-dev-kilt.jsonl",
        )
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--lr", type=float, default=3e-5)
        parser.add_argument("--max_length", type=int, default=32)
        parser.add_argument("--weight_decay", type=int, default=0.01)
        parser.add_argument("--total_num_updates", type=int, default=10000)
        parser.add_argument("--warmup_updates", type=int, default=500)
        parser.add_argument("--num_workers", type=int, default=32)

        parser.add_argument("--model_name", type=str, default="bert-base-uncased")
        parser.add_argument("--eps", type=float, default=0.1)
        return parser

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.model_name)
        self.model = BertClassifier(self.hparams.model_name)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def train_dataloader(self, shuffle=True):
        if not hasattr(self, "train_dataset"):
            self.train_dataset = BinaryKILT(
                tokenizer=self.tokenizer,
                data_path=self.hparams.train_data_path,
                max_length=self.hparams.max_length,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )

    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            self.val_dataset = BinaryKILT(
                tokenizer=self.tokenizer,
                data_path=self.hparams.dev_data_path,
                max_length=self.hparams.max_length,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def forward(self, batch):
        return self.model(
            input_ids=batch["src_input_ids"],
            attention_mask=batch["src_attention_mask"],
        )

    def training_step(self, batch, batch_idx=None):
        logits = self.model(batch)

        cr = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            batch["labels"],
        )

        entropy = torch.distributions.Bernoulli(logits=logits).entropy().mean(-1)

        loss = cr - self.hparams.eps * entropy

        self.log("cr", cr, on_step=True, on_epoch=False, prog_bar=True)
        self.log("entropy", entropy, on_step=True, on_epoch=False, prog_bar=True)
        self.train_acc(logits.sigmoid(), batch["labels"].long())
        self.log(
            "train_acc", self.train_acc, on_step=True, on_epoch=False, prog_bar=True
        )

        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx=None):
        logits = self.model(
            input_ids=batch["src_input_ids"],
            attention_mask=batch["src_attention_mask"],
        )

        self.valid_acc(logits.sigmoid(), batch["labels"].long())
        self.log(
            "valid_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"logits": logits}

    def sample(self, sentences, **kwargs):
        with torch.no_grad():
            return self.model(
                **{
                    k: v.to(self.device)
                    for k, v in self.tokenizer(
                        sentences,
                        return_tensors="pt",
                        padding=True,
                        max_length=self.hparams.max_length,
                        truncation=True,
                    ).items()
                }
            )

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_updates,
            num_training_steps=self.hparams.total_num_updates,
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]
