CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_task_type ON experiments(task_type);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at);

CREATE INDEX IF NOT EXISTS idx_model_runs_experiment_id ON model_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_model_runs_status ON model_runs(status);
CREATE INDEX IF NOT EXISTS idx_model_runs_started_at ON model_runs(started_at);

CREATE INDEX IF NOT EXISTS idx_performance_logs_model_id ON performance_logs(model_id);
CREATE INDEX IF NOT EXISTS idx_performance_logs_timestamp ON performance_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_logs_metric_name ON performance_logs(metric_name);

CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at);
CREATE INDEX IF NOT EXISTS idx_datasets_file_type ON datasets(file_type);

CREATE INDEX IF NOT EXISTS idx_data_pipelines_status ON data_pipelines(status);
CREATE INDEX IF NOT EXISTS idx_data_pipelines_created_at ON data_pipelines(created_at);
