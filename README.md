# POD for Accellerrated NODE


 ![](https://img.shields.io/static/v1?label=python&message=v3.8.8&color=green&style=plastic)
 ![](https://img.shields.io/static/v1?label=repo%20size&message=1.5%20GB&color=orange&style=plastic)

[commet]: <> (TODO: Update the above repo size at launch to exclude output and data directores. Current estimated size ~100 MB)

## Installation
To ensure all necessary packages are installed consider running the following command. This will not update or adjust the version of any currently installed packages.
```bash
pip install -r .\requirements.txt
```

***

To generate the euler equation dataset run the following

```bash
python data/ee_generator.py
```

## Main Methods 

To recover any original data output run the following commands. All commands default to the VKS data sets.
```bash
python src/hbnode.py
python src/node.py
python src/vae.py
```
Append the command line argument `-h` for assistance, or to learn about any of the passable arguments to the method of interest.

A self running version of these methods can be found in `.\bin\`. These bash scripts will automatically regenerate all of the output data. In addition they contain ideal parameter tuning for particular use cases.

## Data Sets

### VKS

Von Karman flow stream dataset.

### Euler Equations

Euler equation traffic flow dataset. The dataset can be regenerated using the command

```bash
python data/ee_generator.py
```
and the inital conditions may be modified within that file.

***

When exectuing one of the main methods provide the alternative following commandline arguments.

*Note: it is recommended to replace [METHOD] with the given method for example EE_HBNODE*

```bash
python src/hbnode.py --dataset EE --data_dir data/EulerEqs.npz --out_dir out/EE_[METHOD]
```

# Customization

## Directories

    data/ - contains all relevant data files

    doc/ - contains detailed explanations of datasets and/or methods

    lib/ - contains all utilities used by main methods organized by method

    out/ - contains all output files organized by dataset_method

    src/ - contains all main methods

    test/ - contains all files that may test over multiple parameters and/or methods


***
**Credits**

Akil Narayang

Bao Wang
 
Elena Cherkaev
  
Eric Thomas
   
Justin Baker