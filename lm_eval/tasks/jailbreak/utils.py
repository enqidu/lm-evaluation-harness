import datasets




def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _label_to_int(s):
        if s == 'Choice A':
            return 1
        elif s == 'Choice B':
            return 2
        elif s == 'Choice C':
            return 3
        elif s == 'Choice D':
            return 4
        
    
    def _process_doc(doc):
        gold = _label_to_int(doc["Label"])
        instruction = f"""
### Context: {doc["Instructions"]}
### Question: {doc["User"]} {doc["Question"]}
### Options:
(1) {doc['Choice A']}\n(2) {doc["Choice B"]}\n(3) {doc["Choice C"]}\n(4) {doc['Choice D']}\n
### Answer: The correct answer to the given problem is """

        out_doc = {
            "question": instruction,
            "choices": ["(1)", "(2)", "(3)", "(4)"],
            "gold": gold - 1,
        }
        return out_doc

    return dataset.map(_process_doc)
