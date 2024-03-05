# NER-classifier-DistilBERT

This repository implement a Name Entity Classifier using a DistilBERT pretrained model from Hugging Face.

The following process has been implemented for create the NER model.

1. Data Analisis Entities
2. New Dataset Creation
3. Finetunning distilbert-base-cased model 
4. Evaluation of the model
5. Export to ONNX for interoperability
6. Comparison of HF Finetunned model and ONNX model


## 1. Data Analisis Entities

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

## 2. New Dataset Creation

New dataset was created considering these 4 entities (41, 87, 34, 37), samples which does not contains these tags were filtered. All others entities tags were set to zero in the new dataset. 

```python

DatasetDict({
    train: Dataset({
        features: ['id', 'tokens', 'ner_tags'],
        num_rows: 900384
    })
    validation: Dataset({
        features: ['id', 'tokens', 'ner_tags'],
        num_rows: 112494
    })
    test: Dataset({
        features: ['id', 'tokens', 'ner_tags'],
        num_rows: 108378
    })
})
```

## 3. Finetunning distilbert-base-cased model 

The steps followed for finetune the distilbert-base-cased model are the following.
    - Tokenize the samples using sentence piece
    - Pad ner_tags according to the tokenized word
    - Select proper hyperparameters 
    - Set data collator that will dynamically pad the inputs received, as well as the labels
    - Finetune the model with transformer Trainer class
    - Verify the metrics output in the validation test and repeat the process until find a good hyperparameters
    - Save the model and push to HuggingFace Hub

```python

Metrics obtained during finetunning model.
#TODO

```

## 4. Evaluation of the model

The evaluation of the model in the test set produced good performance.

```python

#TODO: replace
{'precision': 0.34146341463414637,
 'recall': 0.34146341463414637,
 'f1': 0.34146341463414637,
 'accuracy': 0.9084967320261438}

```

The confusion matrix obtained is presented in the table.

![Confusion Matrix](./docs/confusion-matrix.png?  "Title")


## 5. Export to ONNX for interoperability

The Finetunned HuggingFace model was exported to ONNX for interoperability and accelerated inference. Below the parameters used, it is important to consider the sizes of the input.

```python

torch.onnx.export(hf_finetunned_model,                                         # model being run
                (sample_input_ids_padded, sample_attention_mask_padded),       # model input (or a tuple for multiple inputs)
                save_onnx_model_directory + 'model.onnx',                      # where to save the model (can be a file or file-like object)
                export_params=True,                                            # store the trained parameter weights inside the model file
                opset_version=10,                                              # the ONNX version to export the model to
                do_constant_folding=True,                                      # whether to execute constant folding for optimization
                input_names = ['input_ids', 'attention_mask'],                 # the model's input names
                output_names = ['logits'],                                     # the model's output names
                dynamic_axes={'input' : {0 : 'batch_size'},                    # variable length axes
                              'output' : {0 : 'batch_size'}})

```

## 6. Comparison of HF Finetunned model and ONNX model

It was verified that both models HF Finetunned model and ONNX model has the same predictions (verified with a sample sanity check logits comparison). Additionaly the metrics evaluation on the test set (precision, recall, f1, accurary) are the same. 

```python
## TODO: metrics

```

In relation to the model inference performance, it was verified that the ONNX optimized model is faster than the Finetunned HF model.














