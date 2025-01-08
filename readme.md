# Industrial CNN Pipeline demo


## Introduction 

This pipeline was created to demonstrate the capabilities of Kubeflow Pipelines (KFP). It is based on the medium post by https://towardsdatascience.com/building-a-vision-inspection-cnn-for-an-industrial-application-138936d7a34a.


## Getting started


### Deploy Kubeflow
Follow the steps found by Barteus to setup Kubeflow with or without COS. Find the instructions here: https://github.com/Barteus/demo-aws-mk8s-ckf-mlflow/tree/main?tab=readme-ov-file#infrastructure-installation

### Deploy or create an S3 endpoint

The demo requires a S3 endpoint to be available. To create this deployment, there are a few options - 

- Minio
- AWS S3
- Other S3 compatible storage solution

For this demo, a combination of Minio and S3 compatible was utilised, although - any S3 endpoint will be acceptable.

### Pipeline deployment


There are two ways to deploy the pipeline, either run the pipeline notebook, or utilise the included pipeline.yaml.

Using the pipeline.yaml - upload this to the KFP UI

### Demo UI

The Demo UI utilises Gradio to run inference on the created CNN model. To run this model, simply install gradio and run the application;

```
activate gradio
python demo.py
```


# Improvements

- Swap out medium post for another model, e.g. PaDiM.
- Automate UI/model updates from Kubeflow pipeline
