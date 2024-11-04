import datasets
import collections
import re
import sys
import unicodedata


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


def doc_to_text(doc):
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]


def doc_to_target(doc):
    return doc["option1"]
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()


def doc_to_choice(doc):
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        answ = 'A' if doc['answer'] == '1' else 'B'
        instruction = f"""
### Question: {doc["sentence"]}
### Options:
A {doc["option1"]}, B {doc["option2"]}
### Answer: The correct answer to the given problem is :"""

        out_doc = {
            "question": instruction,
            "answerKey": answ
        }
        return out_doc

    return dataset.map(_process_doc)
