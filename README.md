# ReID

This is a master thesis by Anton Fredriksson and Björn Elwin at Axis Communications.

We want to improve Re-ID algorithms using Facebook AI's DensePose


## Setup
Clone this alongside the detectron2 repository found [here](https://github.com/facebookresearch/detectron2) so the files are repositories are located next to eachother.

The code is easiest run by creating a virtual environment containing all the required pip installations 

### Setup Environment
Create virtual environment
```console
python3 -m venv env
```
Enter virtual environment 
```console
source env/bin/avtivate
```
Install requirements
```console
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0
 -f https://download.pytorch.org/whl/torch_stable.html matplotlib tqdm tk opencv-python logger pytorch_metric_learning
```
Add path to densepose and detectron files
```console
pip install -e ../detectron2/projects/DensePose  
pip install -e ../detectron2  
```

