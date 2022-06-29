
#### mlflow usage
~~~python
import os
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts
import mlflow

if __name__ == "__main__":
    # Log a parameter (key-value pair)
    log_param("param1", randint(0, 100))

    # Log a metric; metrics can be updated throughout the run
    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)

    # Log an artifact (output file)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")
    log_artifacts("outputs") # to log feature files 
    mlflow.set_tracking_uri("file:///tmp/my_tracking") # if empty, ./mlruns
    mlflow.pytorch.log_model(model, model_path)

# to use a remote tracking server
    import mlflow
    mlflow.set_tracking_uri("http://YOUR-SERVER:4040")
    mlflow.set_experiment("my-experiment")

    # load model
pretrained_model = LightningModule.load_from_checkpoint(PATH)
pretrained_model.freeze()

# use it for finetuning
def forward(self, x):
    features = pretrained_model(x)
    classes = classifier(features)

# or for prediction
out = pretrained_model(x)
api_write({'response': out}

~~~

mlflow ui

and view it at http://localhost:5000.

#### Train
mlflow run sklearn_elasticnet_wine -P alpha=0.5

mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=5.0


#### To start a server. Use a database backed-store-uri to enable model registration. 
 <dialect>+<driver>://<username>:<password>@<host>:<port>/<database>. MLflow supports the database dialects mysql, mssql, sqlite, and postgresql. 
ref: https://www.mlflow.org/docs/latest/tracking.html#storage

#### mlflow server
    --backend-store-uri /mnt/persistent-disk \
    --default-artifact-root s3://my-mlflow-bucket/ \
    --host 0.0.0.0

#### Connecting to a Remote Server

Once you have a server running, simply set MLFLOW_TRACKING_URI to the server’s URI, along with its scheme and port (e.g., http://10.0.0.1:5000). Then you can use mlflow as normal:

~~~python
import mlflow
with mlflow.start_run():
    mlflow.log_metric("a", 1)
~~~


#### client :
~~~python
import mlflow
remote_server_uri = "..." # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)
# Note: on Databricks, the experiment name passed to mlflow_set_experiment must be a
# valid path in the workspace
mlflow.set_experiment("/my-experiment")
with mlflow.start_run():
    mlflow.log_param("a", 1)
    mlflow.log_metric("b", 2)
~~~


git config --global credential.helper store



