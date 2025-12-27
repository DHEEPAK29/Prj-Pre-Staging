# ML Training & Fine-Tuning Tools Checklist

## Experiment Tracking & Logging
- [ ] **MLflow** - Track experiments, log metrics, parameters, and artifacts
- [ ] **Weights & Biases (W&B)** - Comprehensive experiment tracking and visualization
- [ ] **Neptune.ai** - Metadata logging and experiment management
- [ ] **Aim** - Open-source experiment tracker
- [ ] **ClearML** - End-to-end ML experiment management

## Vector & Data Storage
- [ ] **Pinecone** - Managed vector database for embeddings
- [ ] **Weaviate** - Open-source vector search engine
- [ ] **Milvus** - Open-source vector database
- [ ] **Qdrant** - Vector search engine for similarity search
- [ ] **Chroma** - Lightweight vector database

## Model Visualization & Inspection
- [ ] **Netron** - Visualize neural network architecture (.onnx, .pb, .h5, etc.)
- [ ] **TensorBoard** - TensorFlow/PyTorch visualization and profiling
- [ ] **Tensorboard.dev** - Share TensorBoard results online
- [ ] **Wandb Charts** - Create custom visualizations in W&B

## Model & Checkpoints Storage
- [ ] **Hugging Face Hub** - Share and store models
- [ ] **Model Registry (MLflow)** - Version control for models
- [ ] **DVC (Data Version Control)** - Track models and data
- [ ] **Weights & Biases Artifacts** - Store and version models

## Hyperparameter Optimization
- [ ] **Optuna** - Bayesian/tree-structured hyperparameter tuning
- [ ] **Ray Tune** - Distributed hyperparameter tuning
- [ ] **Weights & Biases Sweeps** - Built-in hyperparameter search

## Performance Monitoring
- [ ] **Prometheus** - Metrics collection and monitoring
- [ ] **Grafana** - Visualization dashboards for metrics
- [ ] **ELK Stack** - Log aggregation and analysis

## Data Versioning & Management
- [ ] **DVC** - Version control for data and models
- [ ] **Great Expectations** - Data validation and documentation
- [ ] **Pachyderm** - Data versioning and pipeline orchestration

---

**Quick Setup Reminder:** Start with MLflow/W&B → Log metrics → Visualize with Netron/TensorBoard → Store in vector DB → Monitor with Prometheus/Grafana