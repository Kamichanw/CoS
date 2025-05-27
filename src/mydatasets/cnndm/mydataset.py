import os
import re
from datasets import load_from_disk
import evaluate
from src.mydatasets.dataset_base import DatasetBase


class MyDataset(DatasetBase):
    def __init__(self, size="tiny", use_fewshot=False):
        super().__init__(size=size, use_fewshot=use_fewshot)
        self.dataset = load_from_disk(os.path.join(self.cls_abspath, "dataset"))
        self.references = [data["highlights"] for data in self.dataset]

    def evaluate(self, predictions):
        # predictions = [re.split("\n\nArticle:", p)[0] for p in predictions]
        # rouge = evaluate.load("src/mydatasets/cnndm/rouge")
        # return rouge.compute(predictions=predictions, references=self.references)
        return -1


if __name__ == "__main__":
    dataset = MyDataset(size="tiny")
