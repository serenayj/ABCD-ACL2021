# README 


## Intro
This is a folder for running ABCD parser. 

## Dependencies
python 3.6, pytorch 1.6.0, numpy, nltk (word tokenize and sent tokenze), networkx, dgl, pickle, torchtext. 

## Steps

### Step0: Download pretrained models from [Link](https://drive.google.com/file/d/146NQ9vx5GOcHn1geGI-WgjGEJ-RE5w-4/view?usp=sharing) to this code package 

### Step1: Run your data through Stanford CoreNLP (new version 4.20)[Link](https://stanfordnlp.github.io/CoreNLP/index.html) 
Output is a ".out file". E.g., the input file is "test.complex", then output file is "test.complex.out". Input file should have one sentence per line. 

### Step 2: Run your CoreNLP output data with a preprocessor:
``
python process_test.py 
``
Output is a pickle file, with sentence ids as keys, preprocssed graph as values. You could change the filename and data directory in line 23-24 (variable batch_id and data_path). 

### Step 3: Run the ABCD parser with pretrained models. 
```
python test.py 
```
Remember to modify root_dir (code directory) and glove_dir (where you store glove.6B.100d.txt). Output is a pickle file storing a output dictionary where the keys are sentence indices and values are predicted strings. 

