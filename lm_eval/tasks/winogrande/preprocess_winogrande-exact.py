import datasets
import collections
import re
import sys
import unicodedata



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
### Answer: The correct answer to the given problem is """

        out_doc = {
            "question": instruction,
            "answerKey": answ
        }
        return out_doc

    return dataset.map(_process_doc)
