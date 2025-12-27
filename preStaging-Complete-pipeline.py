import mlflow
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
from optuna.samplers import TPESampler
import prometheus_client
from prometheus_client import Counter, Histogram
import json
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from dagster import job, op
from sacred import Experiment
import neptune.new as neptune
from clearml import Task
from aim import Run
from comet_ml import Experiment as CometExperiment
import requests
from elasticsearch import Elasticsearch
import python_pachyderm

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MLFLOW
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_mlflow(experiment_name):
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    print("  MLflow started")

def log_mlflow_params(params):
    for key, value in params.items():
        mlflow.log_param(key, value)

def log_mlflow_metrics(metrics, step):
    for key, value in metrics.items():
        mlflow.log_metric(key, value, step=step)

def log_mlflow_artifact(artifact_path):
    mlflow.log_artifact(artifact_path)

def register_mlflow_model(model_uri, model_name):
    mlflow.register_model(model_uri, model_name)
    print(f"  Model {model_name} registered")

def transition_mlflow_model(model_name, version, stage):
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(model_name, version, stage)
    print(f"  Model transitioned to {stage}")

def load_mlflow_model(model_uri):
    model = mlflow.pytorch.load_model(model_uri)
    return model

def end_mlflow():
    mlflow.end_run()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# WEIGHTS & BIASES
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_wandb(project_name, experiment_name, config):
    wandb.init(project=project_name, name=experiment_name, config=config)
    print("  W&B initialized")

def log_wandb_metrics(metrics):
    wandb.log(metrics)

def log_wandb_artifact(artifact_path):
    wandb.save(artifact_path)

def setup_wandb_sweep(config, project_name):
    sweep_id = wandb.sweep(config, project=project_name)
    print(f"  W&B sweep created")
    return sweep_id

def run_wandb_sweep(sweep_id, training_func, count=10):
    wandb.agent(sweep_id, function=training_func, count=count)
    print(f"  W&B sweep executed")

def end_wandb():
    wandb.finish()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PINECONE
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_pinecone(api_key, index_name):
    from pinecone import Pinecone
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    print("  Pinecone initialized")
    return index

def upsert_to_pinecone(index, vectors, ids, metadata=None):
    data = [(ids[i], vectors[i], metadata[i] if metadata else {}) for i in range(len(ids))]
    index.upsert(vectors=data)
    print(f"  Upserted {len(ids)} vectors to Pinecone")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# WEAVIATE
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_weaviate(url):
    import weaviate
    client = weaviate.Client(url)
    print("  Weaviate initialized")
    return client

def create_weaviate_class(client, class_name, properties):
    class_obj = {"class": class_name, "properties": properties}
    client.schema.create_class(class_obj)
    print(f"  Weaviate class {class_name} created")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MILVUS
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_milvus(host="localhost", port=##):
    from pymilvus import connections
    connections.connect("default", host=host, port=port)
    print("  Milvus connected")

def create_milvus_collection(collection_name, dim):
    from pymilvus import CollectionSchema, FieldSchema, DataType, Collection
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields)
    collection = Collection(collection_name, schema)
    print(f"  Milvus collection {collection_name} created")
    return collection

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# QDRANT
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_qdrant(url="http://localhost:a"):
    from qdrant_client import QdrantClient
    client = QdrantClient(url)
    print("  Qdrant initialized")
    return client

def create_qdrant_collection(client, collection_name, vector_size):
    from qdrant_client.models import Distance, VectorParams
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"  Qdrant collection created")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CHROMA
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_chroma(persist_directory="./chroma_data"):
    import chromadb
    client = chromadb.Client()
    print("  Chroma initialized")
    return client

def create_chroma_collection(client, collection_name):
    collection = client.create_collection(name=collection_name)
    print(f"  Chroma collection created")
    return collection

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NETRON
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def export_to_netron(model, input_shape, onnx_path="model.onnx"):
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(model, dummy_input, onnx_path, verbose=False,
                     input_names=['input'], output_names=['output'])
    print(f"  Model exported to {onnx_path}")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# TENSORBOARD
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_tensorboard(log_dir="./runs"):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)
    print("  TensorBoard initialized")
    return writer

def log_tensorboard_metrics(writer, metrics, step):
    for key, value in metrics.items():
        writer.add_scalar(key, value, step)

def log_tensorboard_graph(writer, model, input_shape):
    dummy_input = torch.randn(*input_shape)
    writer.add_graph(model, dummy_input)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# OPTUNA
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def create_optuna_study(direction='maximize'):
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction=direction, sampler=sampler)
    print("  Optuna study created")
    return study

def optimize_optuna(study, objective_func, n_trials=50):
    study.optimize(objective_func, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best params: {study.best_params}")
    return study.best_params

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# RAY TUNE
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_ray_tune():
    from ray import tune
    print("  Ray Tune initialized")
    return tune

def run_ray_tune_search(trainable, config, num_samples=10):
    from ray import tune
    results = tune.run(trainable, config=config, num_samples=num_samples)
    print("  Ray Tune search complete")
    return results

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# HUGGING FACE HUB
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def upload_to_huggingface(model, model_name, token):
    from huggingface_hub import login
    login(token=token)
    model.push_to_hub(model_name)
    print(f"  Model uploaded to Hugging Face")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# DVC
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_dvc():
    import subprocess
    subprocess.run(["dvc", "init"], check=True)
    print("  DVC initialized")

def add_to_dvc(file_path):
    import subprocess
    subprocess.run(["dvc", "add", file_path], check=True)
    print(f"  Added {file_path} to DVC")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# GREAT EXPECTATIONS
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_great_expectations():
    from great_expectations.dataset import PandasDataset
    print("  Great Expectations initialized")
    return PandasDataset

def validate_data(dataset, expectations):
    result = dataset.validate(expectations)
    print(f"  Data validation complete")
    return result

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PROMETHEUS
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_prometheus_metrics():
    train_loss = Histogram('train_loss', 'Training loss')
    val_accuracy = Histogram('val_accuracy', 'Validation accuracy')
    print("  Prometheus metrics initialized")
    return train_loss, val_accuracy

def record_prometheus_metric(metric, value):
    metric.observe(value)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# AIRFLOW
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_airflow_dag(dag_id, default_args):
    dag = DAG(dag_id, default_args=default_args, schedule_interval='@daily')
    print("  Airflow DAG initialized")
    return dag

def add_airflow_task(dag, task_id, python_callable):
    task = PythonOperator(task_id=task_id, python_callable=python_callable, dag=dag)
    print(f"  Airflow task added")
    return task

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# DAGSTER
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_dagster_job():
    @job
    def ml_pipeline():
        pass
    print("  Dagster job initialized")
    return ml_pipeline

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SACRED
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_sacred(experiment_name):
    ex = Experiment(experiment_name)
    print("  Sacred experiment initialized")
    return ex

def log_sacred_metrics(ex, metrics):
    for key, value in metrics.items():
        ex.log_scalar(key, value)
    print("  Sacred metrics logged")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NEPTUNE.AI
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_neptune(project, api_token):
    run = neptune.init_run(project=project, api_token=api_token)
    print("  Neptune.ai initialized")
    return run

def log_neptune_metrics(run, metrics):
    for key, value in metrics.items():
        run[f"metrics/{key}"] = value
    print("  Neptune metrics logged")

def log_neptune_artifact(run, artifact_path):
    run["artifacts"].upload(artifact_path)
    print(f"  Artifact uploaded to Neptune")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLEARML
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_clearml(project_name, task_name):
    task = Task.init(project_name=project_name, task_name=task_name)
    print("  ClearML task initialized")
    return task

def log_clearml_metrics(task, metrics, step):
    for key, value in metrics.items():
        task.get_logger().report_scalar("training", key, value, step)
    print("  ClearML metrics logged")

def log_clearml_artifact(task, artifact_path):
    task.upload_artifact(name="model", artifact_object=artifact_path)
    print(f"  Artifact uploaded to ClearML")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# AIM
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_aim(experiment):
    run = Run(experiment=experiment)
    print("  Aim experiment initialized")
    return run

def log_aim_metrics(run, metrics, step):
    for key, value in metrics.items():
        run.track(value, name=key, step=step)
    print("  Aim metrics logged")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# COMET
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_comet(api_key, project_name, workspace):
    experiment = CometExperiment(api_key=api_key, project_name=project_name, workspace=workspace)
    print("  Comet experiment initialized")
    return experiment

def log_comet_metrics(experiment, metrics, step):
    for key, value in metrics.items():
        experiment.log_metric(key, value, step=step)
    print("  Comet metrics logged")

def log_comet_artifact(experiment, artifact_path):
    experiment.log_asset(artifact_path)
    print(f"  Artifact uploaded to Comet")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# GRAFANA
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_grafana(grafana_url, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    print("  Grafana connected")
    return grafana_url, headers

def create_grafana_dashboard(grafana_url, headers, dashboard_json):
    response = requests.post(f"{grafana_url}/api/dashboards/db", 
                            json=dashboard_json, headers=headers)
    print(f"  Grafana dashboard created")
    return response.json()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ELK STACK
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_elk(es_host="localhost", es_port=a):
    es = Elasticsearch([{"host": es_host, "port": es_port}])
    print("  ELK Stack connected")
    return es

def log_to_elk(es, index_name, log_data):
    es.index(index=index_name, doc_type="_doc", body=log_data)
    print(f"  Log indexed in Elasticsearch")

def query_elk(es, index_name, query):
    results = es.search(index=index_name, body=query)
    print(f"  Elasticsearch query executed")
    return results

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PACHYDERM
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def setup_pachyderm(host="localhost", port=a):
    client = python_pachyderm.PfsClient(host=host, port=port)
    print("  Pachyderm connected")
    return client

def create_pachyderm_pipeline(client, pipeline_spec):
    client.create_pipeline(pipeline_spec)
    print("  Pachyderm pipeline created")

def list_pachyderm_repos(client):
    repos = client.list_repo()
    print(f"  Pachyderm repos listed")
    return repos

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MODEL
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class SimpleModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# TRAINING
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def train_model(model, train_loader, val_loader, optimizer, criterion, 
                epochs=10, device='cpu', mlflow_log=True, wandb_log=True, 
                tb_writer=None, prometheus_metrics=None):
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        val_acc = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == batch_y).float().mean().item()
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        if mlflow_log:
            log_mlflow_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc}, epoch)
        
        if wandb_log:
            log_wandb_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc, "epoch": epoch})
        
        if tb_writer:
            log_tensorboard_metrics(tb_writer, {"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc}, epoch)
        
        if prometheus_metrics:
            record_prometheus_metric(prometheus_metrics[0], train_loss)
            record_prometheus_metric(prometheus_metrics[1], val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SAVE MODEL
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def save_model(model, model_path="model.pt", mlflow_log=True, wandb_log=True, dvc_log=False):
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved")
    
    if mlflow_log:
        log_mlflow_artifact(model_path)
    
    if wandb_log:
        log_wandb_artifact(model_path)
    
    if dvc_log:
        add_to_dvc(model_path)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MAIN
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    setup_mlflow("ml-training-demo")
    setup_wandb("ml-training", "demo", {"framework": "pytorch"})
    
    model = SimpleModel(input_size=784, hidden_size=128, output_size=10).to(device)
    export_to_netron(model, (1, 784))
    
    tb_writer = setup_tensorboard()
    prometheus_metrics = setup_prometheus_metrics()
    
    hyperparams = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5,
        "optimizer": "Adam"
    }
    log_mlflow_params(hyperparams)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    
    save_model(model, mlflow_log=True, wandb_log=True)
    
    end_mlflow()
    end_wandb()
    print("  Complete!")

if __name__ == "__main__":
    main()
