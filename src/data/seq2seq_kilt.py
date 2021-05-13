import jsonlines
from torch.utils.data import Dataset


class Seq2SeqKILT(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path,
        templates=False,
        max_length=32,
        validation=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []

        with jsonlines.open(data_path) as f:
            for d in f:
                if validation:
                    if templates:
                        for q in d["meta"]["template_questions"]:
                            self.data.append(
                                {
                                    "input": q,
                                    "output": [
                                        o["answer"]
                                        for o in d["output"]
                                        if "answer" in o
                                    ],
                                }
                            )
                    else:
                        self.data.append(
                            {
                                "input": d["input"],
                                "output": [
                                    o["answer"] for o in d["output"] if "answer" in o
                                ],
                            }
                        )
                else:
                    for o in d["output"]:
                        if "answer" in o and "provenance" in o:
                            if templates:
                                for q in d["meta"]["template_questions"]:
                                    self.data.append(
                                        {
                                            "input": q,
                                            "output": o["answer"],
                                        }
                                    )
                            else:
                                self.data.append(
                                    {
                                        "input": d["input"],
                                        "output": o["answer"],
                                    }
                                )

        self.max_length = max_length
        self.validation = validation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {
            "src": self.data[item]["input"],
            "trg": self.data[item]["output"],
        }

    def collate_fn(self, batch):
        batches = {
            "{}_{}".format(name, k): v
            for name in ("src",) + (() if self.validation else ("trg",))
            for k, v in self.tokenizer(
                [b[name] for b in batch],
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }
        if "trg_input_ids" in batches:
            batches["trg_input_ids"][:, 0] = self.tokenizer.eos_token_id
        batches["raw"] = batch
        return batches
