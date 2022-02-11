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
To generate the KPP dataset run the following

```bash
python data/kpp_generator.py
```

To generate the euler equation dataset run the following

```bash
python data/ee_generator.py
```
The VKS dataset can be found [here](url)

## Executing Experiments

All experiments are set in `./bin/run.sh`. The script takes one commandline argument for your python executable {python,python3}.

```bash
sh ./bin/run.sh python3
```

This script takes ~4hr to execute. All experimental results are broken into separate executable scripts in the `./bin/` directory.

## Executing Code

All main methods are located in the './src/' directory.

All code should be executed from the relative path `../pod_hbnode/` .

All directory references should follow the format `./**/dir/` with leading `./` and ending `/`. The relative path should also be from `../pod_hbnode/'.

All commandline arguments are listed in their respective files. You may type the keywork `--help` for a full listing of supported arguments.

```bash
python3 src/run_vae.py --help
```
