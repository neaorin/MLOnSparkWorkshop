# Databricks notebook source
# MAGIC %md ## Register Model, Create Image and Deploy Service
# MAGIC 
# MAGIC This example shows how to deploy a web service in step-by-step fashion:
# MAGIC 
# MAGIC  1. Register model
# MAGIC  2. Query versions of models and select one to deploy
# MAGIC  3. Create Docker image
# MAGIC  4. Query versions of images
# MAGIC  5. Deploy the image as web service  
# MAGIC  6. Make a few test calls to the webservice

# COMMAND ----------

# MAGIC %md ## Prerequisites
# MAGIC 
# MAGIC You need to have run the **Databricks - Credit Scoring** notebook successfully, and as a result have a completed AutoML Experiment.

# COMMAND ----------

# Check core SDK version number
import azureml.core

print("SDK version:", azureml.core.VERSION)

# COMMAND ----------

# Get the last seven letters of the username which will be used to build up exp name
import re

regexStr = r'^([^@]+)@[^@]+$'
emailStr = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply("user")
matchobj = re.search(regexStr, emailStr)
if not matchobj is None:
    if len(matchobj.group(1)) > 10:
        notebook_username = matchobj.group(1)[-10:]
    else:
        notebook_username = matchobj.group(1)
        
    print(notebook_username)
else:
    print("Did not match")

# COMMAND ----------

# MAGIC %md ## Initialize Workspace
# MAGIC 
# MAGIC Initialize an Azure ML Workspace object.

# COMMAND ----------

from azureml.core import Workspace

subscription_id = "6787a35f-386b-4845-91d1-695f24e0924b" #you should be owner or contributor
resource_group = "GlobalAINight-ML-RG" #you should be owner or contributor
workspace_name = "globalainight-ml-wksp" #your workspace name

ws = Workspace(workspace_name = workspace_name,
               subscription_id = subscription_id,
               resource_group = resource_group)

print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')

# COMMAND ----------

# MAGIC %md ### Register Model

# COMMAND ----------

# MAGIC %md 
# MAGIC We are going to register the model we trained for credit scoring in the previous lesson. 
# MAGIC 
# MAGIC First, we are going to use the Azure ML Service SDK to identify our previously-created experiment.

# COMMAND ----------

experiment_name_prefix = 'automl-scoring-' + notebook_username
experiment = [e for e in azureml.core.Experiment.list(ws) if e.name.startswith(experiment_name_prefix)][-1]
experiment

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Now it's time to retrieve the best model we have identified during the experiment.

# COMMAND ----------

from azureml.train.automl.run import AutoMLRun
run = [AutoMLRun(experiment, r.id) for r in azureml.core.Run.list(experiment, status='Completed', type='automl')][-1]
run

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC It's now time to register the model. 
# MAGIC 
# MAGIC You can add tags and descriptions to your models when you register them. 

# COMMAND ----------

model = run.register_model(model_name = "credit_scoring_" + notebook_username,
                       tags = {'area': "Credit scoring", 'type': "classification"}, 
                       description = "Credit Scoring model",
                       iteration = None, # you can deploy a specific iteration 
                       metric="AUC_weighted") # you can deploy the best model according to a different metric, for example "accuracy"

# COMMAND ----------

# MAGIC %md You can explore the registered models within your workspace and query by tag. Models are versioned. If you call the register_model command many times with same model name, you will get multiple versions of the model with increasing version numbers.

# COMMAND ----------

from azureml.core import Model

my_models = Model.list(workspace=ws, tags=['area'])
for m in my_models:
    print("Name:", m.name,"\tVersion:", m.version, "\tDescription:", m.description, m.tags)

# COMMAND ----------

# MAGIC %md ### Create Docker Image

# COMMAND ----------

# MAGIC %md In addition to the registered model, in order to create a Docker image we will also need:
# MAGIC - a scoring Python file (*score.py*), which will be called whenever there is a request for a prediction 
# MAGIC - a Conda dependencies file (*myenv.yml*), which contains all other Anaconda dependencies which should be included in the image
# MAGIC 
# MAGIC Optionally you can also provide:
# MAGIC - a Dockerfile, if you'd rather [use your own custom Docker image](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-custom-docker-image) instead of the one provided by Azure ML Service
# MAGIC - a schema file, used for generating Swagger specs for a model deployment.
# MAGIC - additional files to include in the image. In this example, we will also include the *myconfig.json* file which will include a setting for the model name to use.

# COMMAND ----------

import json

config = {'model_name' :  "credit_scoring_" + notebook_username}
with open('myconfig.json', 'w') as outfile:
    json.dump(config, outfile)

# COMMAND ----------

# MAGIC %%writefile score.py
# MAGIC import json
# MAGIC import pickle
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import azureml.train.automl
# MAGIC from sklearn.externals import joblib
# MAGIC from azureml.core.model import Model
# MAGIC 
# MAGIC from inference_schema.schema_decorators import input_schema, output_schema
# MAGIC from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
# MAGIC from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
# MAGIC 
# MAGIC 
# MAGIC input_sample = pd.DataFrame(data=[{"RevolvingUtilizationOfUnsecuredLines":"0.76","NumberOfTime30-59DaysPastDueNotWorse":"2","DebtRatio":"0.80","NumberOfOpenCreditLinesAndLoans":"13","NumberOfTimes90DaysLate":"0","NumberRealEstateLoansOrLines":"6","NumberOfTime60-89DaysPastDueNotWorse":"0","NumberOfDependents":"2","age_banding":"45-49","MonthlyIncome":"9120.0"}])
# MAGIC output_sample = np.array([0])
# MAGIC 
# MAGIC 
# MAGIC def init():
# MAGIC     global model
# MAGIC     with open('myconfig.json', 'r') as file:
# MAGIC       config = json.load(file)
# MAGIC     model_path = Model.get_model_path(model_name = config['model_name'])
# MAGIC     model = joblib.load(model_path)
# MAGIC 
# MAGIC 
# MAGIC @input_schema('data', PandasParameterType(input_sample))
# MAGIC @output_schema(NumpyParameterType(output_sample))
# MAGIC def run(data):
# MAGIC     try:
# MAGIC         result = model.predict(data)
# MAGIC         return result.tolist()
# MAGIC     except Exception as e:
# MAGIC         result = str(e)
# MAGIC         return json.dumps({"error": result})
# MAGIC     return json.dumps({"result": result.tolist()})

# COMMAND ----------

# MAGIC %%writefile myenv.yml
# MAGIC name: project_environment
# MAGIC dependencies:
# MAGIC   # The python interpreter version.
# MAGIC   # Currently Azure ML only supports 3.5.2 and later.
# MAGIC - python=3.6.2
# MAGIC 
# MAGIC - pip:
# MAGIC   - azureml-train-automl==1.0.57
# MAGIC   - inference-schema
# MAGIC - numpy
# MAGIC - pandas
# MAGIC - scikit-learn
# MAGIC - py-xgboost<=0.80
# MAGIC channels:
# MAGIC - conda-forge

# COMMAND ----------

# MAGIC %md Note that following command can take few minutes. 
# MAGIC 
# MAGIC You can add tags and descriptions to images. Also, an image can contain multiple models.

# COMMAND ----------

from azureml.core.image import Image, ContainerImage

image_config = ContainerImage.image_configuration(runtime="python",
                                 execution_script="score.py",
                                 conda_file="myenv.yml",
                                 dependencies=["myconfig.json"],               
                                 tags = {'area': "Credit scoring", 'type': "classification"}, 
                                 description = "Credit Scoring model image"
                                                 )

image = Image.create(name = "credit-scoring-" + notebook_username,
                     # this is the model object. note you can pass in 0-n models via this list-type parameter
                     # in case you need to reference multiple models, or none at all, in your scoring script.
                     models = [model],
                     image_config = image_config, 
                     workspace = ws)

# COMMAND ----------

image.wait_for_creation(show_output = True)

# COMMAND ----------

# MAGIC %md List images by tag and find out the detailed build log for debugging.

# COMMAND ----------

for i in Image.list(workspace = ws,tags = ["area"]):
    print('{}(v.{} [{}]) stored at {} with build log {}'.format(i.name, i.version, i.creation_state, i.image_location, i.image_build_log_uri))

# COMMAND ----------

# MAGIC %md ### Deploy image as web service on Azure Container Instance
# MAGIC 
# MAGIC Note that the service creation can take few minutes.

# COMMAND ----------

from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                               memory_gb = 1, 
                                               tags = {'area': "Credit scoring", 'type': "classification"},
                                               description = 'Predict risk of credit default using Azure AutoML.')

# COMMAND ----------

from azureml.core.webservice import Webservice

aci_service_name = "credit-scoring-" + notebook_username
print(aci_service_name)
aci_service = Webservice.deploy_from_image(deployment_config = aciconfig,
                                           image = image,
                                           name = aci_service_name,
                                           workspace = ws)
aci_service.wait_for_deployment(True)
print(aci_service.state)

# COMMAND ----------

aci_service

# COMMAND ----------

# MAGIC %md ### Test web service

# COMMAND ----------

# MAGIC %md Call the web service with some input data about potential credit applicants to get a prediction.

# COMMAND ----------

new_credit_applicants = [
  [
    {
      "RevolvingUtilizationOfUnsecuredLines": "0.36",
      "NumberOfTime30-59DaysPastDueNotWorse": "0",
      "DebtRatio": "0.56",
      "NumberOfOpenCreditLinesAndLoans": "2",
      "NumberOfTimes90DaysLate": "0",
      "NumberRealEstateLoansOrLines": "1",
      "NumberOfTime60-89DaysPastDueNotWorse": "1",
      "NumberOfDependents": "1",
      "age_banding": "35-39",
      "MonthlyIncome": "1875.0"
    }
  ],
  [
    {
      "RevolvingUtilizationOfUnsecuredLines": "0.77",
      "NumberOfTime30-59DaysPastDueNotWorse": "2",
      "DebtRatio": "0.85",
      "NumberOfOpenCreditLinesAndLoans": "9",
      "NumberOfTimes90DaysLate": "1",
      "NumberRealEstateLoansOrLines": "3",
      "NumberOfTime60-89DaysPastDueNotWorse": "2",
      "NumberOfDependents": "3",
      "age_banding": "45-49",
      "MonthlyIncome": "1010.0"
    }
  ]
]

# COMMAND ----------

import json

test_sample = json.dumps({'data': new_credit_applicants[0]})
test_sample = bytes(test_sample,encoding = 'utf8')

prediction = aci_service.run(input_data=test_sample)
print(prediction)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Now let's try to call the scoring endpoint, like a real business application would do.

# COMMAND ----------

print(aci_service.scoring_uri)

# COMMAND ----------

import urllib.request, urllib.parse

test_sample = json.dumps({'data': new_credit_applicants[1]})
test_sample = bytes(test_sample,encoding = 'utf8')

request = urllib.request.Request(aci_service.scoring_uri);
request.add_header('Content-Type','application/json')
handler = urllib.request.urlopen(request, test_sample)
print( handler.read().decode( 'utf-8' ) );

# COMMAND ----------

# MAGIC %md ## Deploying a web service to Azure Kubernetes Service (AKS)
# MAGIC We can also deploy a model to Azure Kubernetes Service (AKS).
# MAGIC 
# MAGIC [Kubernetes](https://kubernetes.io/) is a portable, extensible, open-source platform for managing containerized workloads and services, that facilitates both declarative configuration and automation. It has a large, rapidly growing ecosystem. Kubernetes services, support, and tools are widely available.
# MAGIC 
# MAGIC 
# MAGIC Kubernetes provides you with [a lot of userful features](https://kubernetes.io/docs/concepts/overview/what-is-kubernetes/#why-you-need-kubernetes-and-what-can-it-do) for your containerized applications:
# MAGIC 
# MAGIC - Service discovery and load balancing
# MAGIC - Storage orchestration
# MAGIC - Automated rollouts and rollbacks
# MAGIC - Automatic bin packing
# MAGIC - Self-healing
# MAGIC - Secret and configuration management
# MAGIC 
# MAGIC [Azure Kubernetes Service (AKS)](https://docs.microsoft.com/en-us/azure/aks/) makes it simple to deploy a managed Kubernetes cluster in Azure. AKS reduces the complexity and operational overhead of managing Kubernetes by offloading much of that responsibility to Azure. As a hosted Kubernetes service, Azure handles critical tasks like health monitoring and maintenance for you. The Kubernetes masters are managed by Azure. You only manage and maintain the agent nodes. As a managed Kubernetes service, AKS is free - you only pay for the agent nodes within your clusters, not for the masters.
# MAGIC 
# MAGIC The steps for deploying a service on AKS are similar: registering a model, creating an image, provisioning a cluster (one time action), and deploying a service to it. We then test and delete the service, image and model.
# MAGIC 
# MAGIC ## Provision the AKS Cluster
# MAGIC 
# MAGIC This is a one time setup. You can reuse this cluster for multiple deployments after it has been created. If you delete the cluster or the resource group that contains it, then you would have to recreate it.
# MAGIC 
# MAGIC > NOTE: For this lab we have already created the AKS cluster as a compute resource attached to Azure ML Service.

# COMMAND ----------

from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import Webservice, AksWebservice

# Use the default configuration (can also provide parameters to customize)
prov_config = AksCompute.provisioning_configuration()

aks_name = 'my-aks-9' 
# Get or create the cluster
aks_target = None

try:
    aks_target = ComputeTarget(workspace=ws, name=aks_name)
    print('Found existing AKS cluster, use it.')
except ComputeTargetException:
    aks_target = ComputeTarget.create(workspace = ws, 
                                  name = aks_name, 
                                  provisioning_configuration = prov_config)
    print('Creating new AKS cluster.')


# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy web service to AKS

# COMMAND ----------

aks_service_name ="aks-credit-scoring-" + notebook_username

aksconfig = AksWebservice.deploy_configuration(cpu_cores = 1, 
                                               memory_gb = 1, 
                                               autoscale_enabled=None, autoscale_min_replicas=None, autoscale_max_replicas=None, autoscale_refresh_seconds=None, 
                                               auth_enabled = True,
                                               primary_key = "fdnmi545k35ogf53434k33",
                                               tags = {'area': "Credit scoring", 'type': "classification"},
                                               description = 'Predict risk of credit default using Azure AutoML.')


aks_service = Webservice.deploy_from_image(deployment_config = aksconfig,
                                           image = image,
                                           name = aks_service_name,
                                           deployment_target = aks_target,
                                           workspace = ws)

aks_service.wait_for_deployment(show_output = True)
print(aks_service.state)

# COMMAND ----------

print(aks_service.scoring_uri)

# COMMAND ----------

import urllib.request, urllib.parse

test_sample = json.dumps({'data': new_credit_applicants[1]})
test_sample = bytes(test_sample,encoding = 'utf8')

request = urllib.request.Request(aks_service.scoring_uri);
request.add_header('Content-Type','application/json')
request.add_header('Authorization','Bearer fdnmi545k35ogf53434k33')
handler = urllib.request.urlopen(request, test_sample)
print( handler.read().decode( 'utf-8' ) );
