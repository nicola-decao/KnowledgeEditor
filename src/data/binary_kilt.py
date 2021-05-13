import jsonlines
import torch
from torch.utils.data import Dataset


class BinaryKILT(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path,
        max_length=32,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []

        with jsonlines.open(data_path) as f:
            for d in f:
                self.data.append(
                    {
                        "input": d["input"],
                        "label": d["output"][0]["answer"],
                    }
                )

        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {
            "src": self.data[item]["input"],
            "label": self.data[item]["label"] == "SUPPORTS",
        }

    def collate_fn(self, batch):
        batches = {
            "{}_{}".format(name, k): v
            for name in ("src",)
            for k, v in self.tokenizer(
                [b[name] for b in batch],
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }
        batches["labels"] = torch.tensor([b["label"] for b in batch]).float()
        batches["raw"] = batch
        return batches
