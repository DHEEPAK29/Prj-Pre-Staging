# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# AIRFLOW
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def setup_airflow_dag(dag_id, default_args):
    dag = DAG(dag_id, default_args=default_args, schedule_interval='@daily')
    print("✓ Airflow DAG initialized")
    return dag

def add_airflow_task(dag, task_id, python_callable):
    task = PythonOperator(task_id=task_id, python_callable=python_callable, dag=dag)
    print(f"✓ Airflow task {task_id} added")
    return task

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# DAGSTER
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from dagster import job, op

def setup_dagster_job():
    @job
    def ml_pipeline():
        pass
    print("✓ Dagster job initialized")
    return ml_pipeline

@op
def dagster_op_example():
    print("✓ Dagster op created")
    return "done"

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SACRED
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from sacred import Experiment

def setup_sacred():
    ex = Experiment("ml-training")
    print("✓ Sacred experiment initialized")
    return ex

@ex.config
def sacred_config():
    learning_rate = 0.001
    batch_size = 32
    epochs = 10

def log_sacred_metrics(ex, metrics):
    for key, value in metrics.items():
        ex.log_scalar(key, value)
    print("✓ Sacred metrics logged")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NEPTUNE.AI
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import neptune.new as neptune

def setup_neptune(project, api_token):
    run = neptune.init_run(project=project, api_token=api_token)
    print("✓ Neptune.ai initialized")
    return run

def log_neptune_metrics(run, metrics):
    for key, value in metrics.items():
        run[f"metrics/{key}"] = value
    print("✓ Neptune metrics logged")

def log_neptune_artifact(run, artifact_path):
    run["artifacts"].upload(artifact_path)
    print(f"✓ Artifact uploaded to Neptune")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLEARML
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from clearml import Task

def setup_clearml(project_name, task_name):
    task = Task.init(project_name=project_name, task_name=task_name)
    print("✓ ClearML task initialized")
    return task

def log_clearml_metrics(task, metrics, step):
    for key, value in metrics.items():
        task.get_logger().report_scalar("training", key, value, step)
    print("✓ ClearML metrics logged")

def log_clearml_artifact(task, artifact_path):
    task.upload_artifact(name="model", artifact_object=artifact_path)
    print(f"✓ Artifact uploaded to ClearML")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# AIM
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from aim import Run

def setup_aim(experiment):
    run = Run(experiment=experiment)
    print("✓ Aim experiment initialized")
    return run

def log_aim_metrics(run, metrics, step):
    for key, value in metrics.items():
        run.track(value, name=key, step=step)
    print("✓ Aim metrics logged")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# COMET
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from comet_ml import Experiment

def setup_comet(api_key, project_name, workspace):
    experiment = Experiment(api_key=api_key, project_name=project_name, workspace=workspace)
    print("✓ Comet experiment initialized")
    return experiment

def log_comet_metrics(experiment, metrics, step):
    for key, value in metrics.items():
        experiment.log_metric(key, value, step=step)
    print("✓ Comet metrics logged")

def log_comet_artifact(experiment, artifact_path):
    experiment.log_asset(artifact_path)
    print(f"✓ Artifact uploaded to Comet")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MLFLOW MODEL REGISTRY
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import mlflow.pytorch

def register_mlflow_model(model_uri, model_name):
    mlflow.register_model(model_uri, model_name)
    print(f"✓ Model {model_name} registered in MLflow")

def transition_mlflow_model(model_name, version, stage):
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(model_name, version, stage)
    print(f"✓ Model {model_name} v{version} transitioned to {stage}")

def load_mlflow_model(model_uri):
    model = mlflow.pytorch.load_model(model_uri)
    print(f"✓ Model loaded from {model_uri}")
    return model

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# WANDB SWEEPS
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import wandb

def setup_wandb_sweep(config, project_name):
    sweep_id = wandb.sweep(config, project=project_name)
    print(f"✓ W&B sweep created: {sweep_id}")
    return sweep_id

def run_wandb_sweep(sweep_id, training_func, count=10):
    wandb.agent(sweep_id, function=training_func, count=count)
    print(f"✓ W&B sweep {sweep_id} executed")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# GRAFANA
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import requests

def setup_grafana(grafana_url, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    print("✓ Grafana connected")
    return grafana_url, headers

def create_grafana_dashboard(grafana_url, headers, dashboard_json):
    response = requests.post(f"{grafana_url}/api/dashboards/db", 
                            json=dashboard_json, headers=headers)
    print(f"✓ Grafana dashboard created")
    return response.json()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ELK STACK
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from elasticsearch import Elasticsearch
import logging

def setup_elk(es_host="localhost", es_port=9200):
    es = Elasticsearch([{"host": es_host, "port": es_port}])
    print("✓ ELK Stack (Elasticsearch) connected")
    return es

def log_to_elk(es, index_name, log_data):
    es.index(index=index_name, doc_type="_doc", body=log_data)
    print(f"✓ Log indexed in Elasticsearch: {index_name}")

def query_elk(es, index_name, query):
    results = es.search(index=index_name, body=query)
    print(f"✓ Elasticsearch query executed")
    return results

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PACHYDERM
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import python_pachyderm

def setup_pachyderm(host="localhost", port=30650):
    client = python_pachyderm.PfsClient(host=host, port=port)
    print("✓ Pachyderm connected")
    return client

def create_pachyderm_pipeline(client, pipeline_spec):
    client.create_pipeline(pipeline_spec)
    print("✓ Pachyderm pipeline created")

def list_pachyderm_repos(client):
    repos = client.list_repo()
    print(f"✓ Pachyderm repos: {repos}")
    return repos
