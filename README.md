# BRIDGE
This is a replication package for `BRIDGE: Enhancing Code Comment Generation with Bytecode CFG Graph Encoding and Retrieval-Augmented DECOME`. 
Our project is public at: <>

## Content
1. [Get Started](#1-Get-Started)<br>
&ensp;&ensp;[1.1 Requirements](#11-Requirements)<br>
&ensp;&ensp;[1.2 Dataset](#12-Dataset)<br>
&ensp;&ensp;[1.3 Train and Test](#13-Train-and-Test)<br>

## 1 Get Started
### 1.1 Requirements
* Hardwares: NVIDIA GeForce RTX 3060 GPU, intel core i5 CPU
* OS: Ubuntu 20.04
* Packages: 
  * python 3.8 (for running the main code)
  * pytorch 1.9.0
  * cuda 11.1
  * java 1.8.0 (for retrieving the similar code)
  * python 2.7 (for evaluation)

### 1.2 Dataset
BRIDGE is evaluated on [JCSD](https://github.com/sdfdfx/TSE) benchmark datasets. The structures of ```dataset/JCSD``` are as follows:
* train/valid/test
  *  source.code: tokens of the source code
  *  source.comment: tokens of the source comments
  *  source.keywords: tokens of the code keywords(identifier names)
  *  similar.code: tokens of the sourcecode retrieved similar code
  *  similar.sourcecomment: tokens of the retrieved similar comments 
  *  similar.bytecode: Similar bytecodes retrieved using bytecodes
  *  similar.bytecomment: tokens of the bytecodes retrieved similar comments 


### 1.3 Train and Test
1. Go the ```src``` directory, process the dataset and generate the vocabulary:
```
cd src/
python build_vocab.py
```
2. Train DECOM model by performing a two-step training strategy:
```
python train_locally.py
python train_FineTune.py
```
3. Test DECOM model:
```
python prediction.py
```
4. Switch to python 2.7 environment and evaluate the performance of DECOM:
```
cd rencos_evaluation/
python evaluate.py
```
