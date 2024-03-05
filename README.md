# NER-classifier-DistilBERT

This repository implement a Name Entity Classifier using a DistilBERT pretrained model from Hugging Face.

The following process has been implemented for create the NER model.

1. Data Analisis Entities
2. New Dataset Creation
3. Finetunning distilbert-base-cased model 
4. Evaluation of the model
5. Export to ONNX for interoperability
6. Comparison of HF Finetunned model and ONNX model


## Data Analisis Entities

The finer-139 dataset contains 3 sets.

train: 900384 samples
test: 108378 samples
validation: 112494 samples

The most repeated entities for the entire dataset are 0, 41, 87, 34, 37.

```python

{   'O': 0,
    'B-DebtInstrumentInterestRateStatedPercentage': 41,
    'B-LineOfCreditFacilityMaximumBorrowingCapacity': 87,
    'B-DebtInstrumentBasisSpreadOnVariableRate1': 34,
    'B-DebtInstrumentFaceAmount': 37
 }

```

0  : 50755070 repetitions
41 : 18448 repetitions
87 : 14730 repetitions
34 : 14469 repetitions
37 :13158 repetitions

![Entities distribution](./docs/entities-distribution.png?  "Title")




