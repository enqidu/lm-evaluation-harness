import datasets




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
### Question: {doc["Question"]}
### Options:
A {doc['A']}\n(B {doc["B"]}\nC {doc["C"]}\nD {doc['D']}\n
### Answer: The correct answer to the given problem is """

        out_doc = {
            "question": instruction,
            "choices": ["A", "B", "C", "D"],
            "gold": gold - 1,
            "label": doc["Label"]
        }
        return out_doc

    return dataset.map(_process_doc)
