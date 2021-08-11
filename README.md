
# Classifier Api

Binary classifer Api for classifing a subset of CIFAR10 Dataset

## API Reference


#### Predict Category

```http
POST /predict
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `file`      | `file` | **Required**. Image to be predicted |


  
## Run Locally

Clone the project

```bash
  git clone https://github.com/abhinav-TB/Classifier-Api.git
```

Go to the project directory

```bash
  cd Classifier-Api
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  python3 app/main.py
```

  
## Testing Endpoints

```bash
  python3 test/test.py
```


  
