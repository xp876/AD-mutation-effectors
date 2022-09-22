# AD-mutation-effectors
This is for "AD-Syn-Net: Systematic identification of Alzheimer’s disease-associated mutation and co-mutation vulnerabilities via deep learning".


Python packages information:
```
h5py                      2.10.0
eli5                      0.11.0
tensorflow                2.4.1
scikit-learn              0.23.2
python                    3.8.5
lime                      0.2.0.1
keras                     2.4.3
h5py                      2.10.0
```

For Deep-SMCI and Deep-CMCI models, please refer to https://zenodo.org/record/7102185#.YyuuT3bMKPo.
The complete processed datasets are available if the requesters are approved by Alzheimer's Disease Neuroimaging Initiative: ADNI.
We also provide example datasets in the above link.
To run Deep-SMCI and Deep-CMCI, please format your input files such as mutations matrix or co-mutations matrix like the example datasets we provide in the Zenodo link.

Here is demo for running Deep-SMCI and Deep-CMCI:
```
For Deep-SMCI:
Python Predict.py Deep-SMCI.h5 example_mutation.txt AD-score.txt
For Deep-CMCI:
Python Predict.py Deep-CMCI.h5 example_co_mutations.txt AD-score.txt

```
To note, AD-score is the prediction output file
