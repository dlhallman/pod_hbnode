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

