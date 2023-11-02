# Sequential Modeling in Marine Navigation: Case Study on a Passenger Vessel
West Coat Vessel - visualization/classification/prediction

Credits:

This project is done in collaboration with Simon Fraser University (SFU) and National Research Council Canada (NRC)

## Contents
1. [Setup](#setup)
2. [Projects](#projects)
3. [Dataset](#dataset)
4. [Gym Environment](#simulator)
5. [Preprocessing](#preprocessing)
6. [Training](#training)
7. [Optimization](#optimization)
8. [How to cite](#cite)

## Setup
Clone the repo, and build the conda environment:

```Shell
git clone git@github.com:pagand/model_optimze_vessel.git
cd model_optimze_vessel
conda create -n vessel python=3.9
conda activate vessel
pip install -r requirements.txt
```

If you have GPU with CUDA enabled:
CUDA<=10.2
```Shell
pip uninstall torch torchvision torchaudio #(run twice)
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102
```

CUDA>10.2
```Shell
pip uninstall torch torchvision torchaudio #(run twice)
pip install torch==1.12.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

Others:
Install the correct version of pytorch given your CUDA from [previous versions](https://pytorch.org/get-started/previous-versions/) or [start locally](https://pytorch.org/get-started/locally/). Replcae the {version} with the correct compatible version.
```Shell
conda install pytorch=={version1} torchvision=={version2} cudatoolkit={version3} -c pytorch
```

Install Huggingface transformers or follow [the link](https://huggingface.co/docs/transformers/installation)

```Shell
pip install transformers
```

To download the RL dataset, please run
```Shell
sh download_data.sh
```

## Projects:

1- (non-)Parameteric modeling of fuel consumption. Accepted in  Ocean Engineering version, please refer to the OE branch.

2- Sequentaioal modelling. Currently in the main branch.

3- Optimization. (TODO)


## Dataset
1- Original data (Confidential). Please put <queenCsvOut.csv> and <queenCsvOutAugmented.csv> in data folder.

2- Offline RL dataset

To download the RL dataset, please run
```Shell
sh download_data.sh
```

This will download the normalized trips in the data folder.


## Simulator
For the gym enviroenmt, please refer to the simulator folder.

## Preprocessing

### Visulization
To get an exploratory data analysis (EDA) of the data, please refer to the visulization folder. 

### Feature selection
Please refer to the Features section to get the insight on the feature engineering and feature selection proccess.

### Prepration
To get the handle missing values, outlier detection, and normalization, please refer to the prepration folder. 



## Training
For the model, we have two approaches.

### Functional model
For this model, please refer to model/functional

### Sequential model
For this model, please refer to the model/sequential

## Cite

```
@inproceedings{Fan2024sequential,
  title={Sequential Modeling in Marine Navigation: Case Study on a Passenger Vessel},
  author={Fan, Yimeng and  Agand, Pedram  and Chen, Mo and Park, Edward J and Kennedy, Allison and Bae, Chanwoo},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={},
  number={},
  pages={},
  year={2024}
}

  @article{agand2023fuel,
  title={Fuel consumption prediction for a passenger ferry using machine learning and in-service data: A comparative study},
  author={Agand, Pedram and Kennedy, Allison and Harris, Trevor and Bae, Chanwoo and Chen, Mo and Park, Edward J},
  journal={Ocean Engineering},
  volume={284},
  pages={115271},
  year={2023},
  publisher={Elsevier}
}
```


