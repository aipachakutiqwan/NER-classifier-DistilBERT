# NER-classifier-DistilBERT

This repository implement a Name Entity Classifier using a DistilBERT pretrained model from Hugging Face.

The finetunning results will be reproduced in Google Colab using T4 GPU, feel free to clone in Colab.

<a href="https://drive.google.com/file/d/1DbbvAdZ5lYEXfIvhMfr91SA5LHWj-yxq/view?usp=share_link" target="_blank">Colab Notebook</a>


Locally reproducible code version use small sample of the entire dataset.

```console
src/notebooks/Locally-NER-Classifier-DistilBERT.ipynb
```

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
    37 : 13158 repetitions

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

The model finetunning run in Google Colab using a single T4 GPU. The steps followed for finetune the distilbert-base-cased model are the following.

    - Tokenize the samples using sentence piece
    - Pad ner_tags according to the tokenized word
    - Select proper hyperparameters 
    - Set data collator that will dynamically pad the inputs received, as well as the labels
    - Finetune the model with transformer Trainer class
    - Verify the metrics output in the validation test and repeat the process until find good hyperparameters
    - Save the model and push to HuggingFace Hub

Final hyperparameters setting.

```python

args = TrainingArguments(finetunned_ner_classifier_distil_bert,
                         evaluation_strategy = "epoch",
                         save_strategy = "epoch",
                         learning_rate = 2e-5,
                         num_train_epochs = 10,
                         weight_decay = 0.01,
                         logging_steps = 50, # default 500,
                        )

```

The overall metrics obtained during finetunning. 

![Confusion Matrix](./docs/training-metrics.png?  "Title")

Training finetunning metrics for every entity.

```python
{'DebtInstrumentBasisSpreadOnVariableRate1':  {'precision': 0.9484620085557139, 
                                               'recall': 0.9677821658698815, 
                                               'f1': 0.9580246913580246, 
                                               'number': 4811 }, 

 'DebtInstrumentFaceAmount':                 {'precision': 0.861864406779661, 
                                               'recall': 0.8905429071803853, 
                                               'f1': 0.875968992248062, 
                                               'number': 3426 }, 

'DebtInstrumentInterestRateStatedPercentage': {'precision': 0.9589780496581504, 
                                               'recall': 0.9615731553310481, 
                                               'f1': 0.9602738492027744, 
                                               'number': 5543 }, 

'LineOfCreditFacilityMaximumBorrowingCapacity': {'precision': 0.9013248058474189, 
                                                 'recall': 0.9153328694038506, 
                                                 'f1': 0.9082748302451376, 
                                                 'number': 4311 }

}
```

Model Finetunned will be downloaded from https://huggingface.co/florenciopaucar/ner-classifier-distil-bert


## 4. Evaluation of the model

The evaluation of the model in the test set produced good performance.

```python

{ 'precision': 0.9087918865209347,
  'recall':    0.9172141918528253,
  'f1':        0.9129836155858461,
  'accuracy':  0.9912687548406852
 }

```
Metrics in the test set for every entity.

```python

{

'DebtInstrumentBasisSpreadOnVariableRate1':    {'precision': 0.9417948717948718, 
                                                 'recall': 0.9449446874196038, 
                                                 'f1': 0.9433671503788366, 
                                                 'number': 3887 }, 
'DebtInstrumentFaceAmount':                    {'precision': 0.8254875588433087, 
                                                 'recall': 0.8480138169257341, 
                                                 'f1': 0.8365990799113989, 
                                                 'number': 2895 }, 
'DebtInstrumentInterestRateStatedPercentage':  {'precision': 0.9369306236860546, 
                                                 'recall': 0.9572792362768496, 
                                                 'f1': 0.9469956321567701, 
                                                 'number': 4190 }, 
'LineOfCreditFacilityMaximumBorrowingCapacity': {'precision': 0.9083769633507853, 
                                                 'recall': 0.8956122741611701, 
                                                 'f1': 0.9019494584837545, 
                                                 'number': 3487 }
}

```

The confusion matrix obtained for the 4 entities are presented. Table rows represent predicted values 
and columns real label values.


    O                                                :  0
    B-DebtInstrumentInterestRateStatedPercentage     : 41
    B-LineOfCreditFacilityMaximumBorrowingCapacity   : 87
    B-DebtInstrumentBasisSpreadOnVariableRate1       : 34
    B-DebtInstrumentFaceAmount                       : 37



![Confusion Matrix](./docs/confusion-matrix.png?  "Title")


## 5. Export to ONNX for interoperability

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

Both models HF Finetunned model and ONNX model has the same predictions (verified with a sample sanity check logits comparison). Additionaly the metrics evaluation on the test set (precision, recall, f1, accurary) for both models are the same. 

There are some preprocessing of the input (padding to 512 tensor size) and postprocessing of the output which take some additional time for onnx model.






