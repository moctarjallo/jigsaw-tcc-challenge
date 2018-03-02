from tcc.api import classifier
from tcc.io import Processor, KModel, CSVData

test_data = CSVData('data/sample_train.csv')
processor = Processor()(test_data)
model = KModel(processor.max_len, 6)

classes = classifier(test_data, processor, model=model)

print(classes)

# Problems:
    # cannot test on new data: Key Error raised because of non existing word in word_to_int dictionary
    # is the choosen vocabulary complete enough?
    # what alternatives?

