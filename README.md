# Federated learning and differential privacy in DL: healthcare applications
Nil Adell Mill

### Short project

In progress...


## Installation

In order to use you need to install [Selene-SDK](https://github.com/FunctionLab/selene) and [PySyft](https://github.com/OpenMined/PySyft). What has worked so far for me in clean installs was to, first, use conda to create a new environment with the env file provided in this repo
```conda env create -f environment.yml```

  , then, install Selene from source
```
git clone https://github.com/FunctionLab/selene.git
cd selene
pip install .
```

 , and finally, install PySyft (tweaked version*)
```
git clone -b dev-quickfix-cuda https://github.com/niladell/PySyft.git
cd PySyft
pip install .
```

_*Disclaimer: Altough this is something that is gonna be solved soon (I hope). As of today the current version of PySyft's federated learning breaks. I added a [dirty-fix here](https://github.com/niladell/PySyft/tree/dev-quickfix-cuda); you can also check [the original issue](https://github.com/OpenMined/PySyft/issues/1893) to see if there's any progress._
