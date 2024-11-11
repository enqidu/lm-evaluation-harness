import datasets
import collections
import re
import sys
import unicodedata



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
                

                filtered_resp.append(resp)

            return filtered_resp

        # Apply the filter set function to each response in resps
        filtered_resps = [filter_set(resp) for resp in resps]

        return filtered_resps



def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
                    
    def _process_doc(doc):
        s = doc["Label"]
        if s == 'A':
            gold =  1
        elif s == 'B':
            gold = 2
        elif s == 'C':
            gold = 3
        elif s == 'D':
            gold = 4
        instruction = f"""
### Instructions: {doc["Instructions"]}
### User : {doc["User"]}
### Expert Opinion: {doc['Expert']} 
### Question: {doc["Question"]}
### Options:
A {doc['A']}\n(B {doc["B"]}\nC {doc["C"]}\nD {doc['D']}\n
### Answer: The correct answer to the given problem is (only letter): """

        out_doc = {
            "question": instruction,
            "choices": ["A", "B", "C", "D"],
            "gold": gold - 1,
            "label": doc["Label"]
        }
        return out_doc

    return dataset.map(_process_doc)