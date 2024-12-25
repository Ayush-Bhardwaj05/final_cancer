# Breast cancer diagnosis predictor



 Public dataset [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). 



## Usage

```bash
pip install -r requirements.txt
```

### Front End

```bash
cd final_cancer/app
```
```bash
conda activate cancer-env
```
```bash
streamlit run main.py
```

### Back End

```bash
cd final_cancer/model
```
```bash
conda activate python37-env
```
```bash
python ensemble_model.py
```
