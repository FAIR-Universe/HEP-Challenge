# Conda

The following steps will help you setup a conda environment.

1. Install Conda
https://conda.io/projects/conda/en/latest/user-guide/install/index.html

2. Create env
```
conda create --name fair python=3.9
```

3. Activate env
```
conda activate fair
```

4. Install required packages
```
pip install -r requirements.txt
```

5. (Optional) Remove env
After you have used the env and you want to remove it, use the following command
```
conda env remove --name fair
```