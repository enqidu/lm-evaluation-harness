import datasets
import collections
import re
import sys
import unicodedata
from lm_eval.filters.extraction import Filter, RegexFilter




from lm_eval.filters.extraction import Filter, RegexFilter


class ParenthesisFilter(Filter):
    """A filter that removes leading parentheses from responses."""

    def __init__(self) -> None:
        """Initializes the ParenthesisFilter."""
        pass

    def apply(self, resps, docs):
        """Applies the filter to remove leading parentheses from each response."""
        def filter_set(inst):
            filtered_resp = []
            for resp in inst:
                # Remove a leading opening or closing parenthesis
                if resp.startswith("(") or resp.startswith(")"):
                    resp = resp[1:]
                if '### Answer:' in resp:
                    resp = resp.split('### Answer:')[1].strip()


                filtered_resp.append(resp)

            return filtered_resp

        # Apply the filter set function to each response in resps
        filtered_resps = [filter_set(resp) for resp in resps]

        return filtered_resps



def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
                    
    def _process_doc(doc):
        s = doc["answerKey"].strip()
        gold = 1
        if s == 'A':
            gold = 1
        elif s == 'B':
            gold = 2
        elif s == 'C':
            gold = 3
        elif s == 'D':
            gold = 4
        choices = " ".join([f"{v} : {k}" for k, v in zip(doc["choices"]["text"], doc["choices"]["label"])])
        instruction = f"""
### Question: {doc["question"]}
### Options:
{choices}
### Answer: The correct answer to the given problem is """

        out_doc = {
            "question": instruction,
            "choices": ["A", "B", "C", "D"],
            "gold": gold - 1,
            "answerKey": doc["answerKey"]
        }
        return out_doc

    return dataset.map(_process_doc)
