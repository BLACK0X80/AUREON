"""Initial migration

Revision ID: 001_initial
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial tables."""
    # Create experiments table
    op.create_table('experiments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('task_type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('config', sa.Text(), nullable=True),
        sa.Column('results', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    
    # Create datasets table
    op.create_table('datasets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=False),
        sa.Column('file_type', sa.String(length=20), nullable=False),
        sa.Column('size', sa.Integer(), nullable=True),
        sa.Column('rows', sa.Integer(), nullable=True),
        sa.Column('columns', sa.Integer(), nullable=True),
        sa.Column('target_column', sa.String(length=100), nullable=True),
        sa.Column('metadata', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create model_runs table
    op.create_table('model_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('experiment_id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('model_type', sa.String(length=50), nullable=False),
        sa.Column('parameters', sa.Text(), nullable=True),
        sa.Column('metrics', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create data_pipelines table
    op.create_table('data_pipelines',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('config', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('input_data_id', sa.Integer(), nullable=True),
        sa.Column('output_data_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create performance_logs table
    op.create_table('performance_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('dataset_id', sa.Integer(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('metadata', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('idx_experiments_status', 'experiments', ['status'])
    op.create_index('idx_experiments_task_type', 'experiments', ['task_type'])
    op.create_index('idx_experiments_created_at', 'experiments', ['created_at'])
    
    op.create_index('idx_model_runs_experiment_id', 'model_runs', ['experiment_id'])
    op.create_index('idx_model_runs_status', 'model_runs', ['status'])
    op.create_index('idx_model_runs_started_at', 'model_runs', ['started_at'])
    
    op.create_index('idx_performance_logs_model_id', 'performance_logs', ['model_id'])
    op.create_index('idx_performance_logs_timestamp', 'performance_logs', ['timestamp'])
    op.create_index('idx_performance_logs_metric_name', 'performance_logs', ['metric_name'])
    
    op.create_index('idx_datasets_created_at', 'datasets', ['created_at'])
    op.create_index('idx_datasets_file_type', 'datasets', ['file_type'])
    
    op.create_index('idx_data_pipelines_status', 'data_pipelines', ['status'])
    op.create_index('idx_data_pipelines_created_at', 'data_pipelines', ['created_at'])


def downgrade() -> None:
    """Drop all tables."""
    op.drop_index('idx_data_pipelines_created_at', table_name='data_pipelines')
    op.drop_index('idx_data_pipelines_status', table_name='data_pipelines')
    op.drop_index('idx_datasets_file_type', table_name='datasets')
    op.drop_index('idx_datasets_created_at', table_name='datasets')
    op.drop_index('idx_performance_logs_metric_name', table_name='performance_logs')
    op.drop_index('idx_performance_logs_timestamp', table_name='performance_logs')
    op.drop_index('idx_performance_logs_model_id', table_name='performance_logs')
    op.drop_index('idx_model_runs_started_at', table_name='model_runs')
    op.drop_index('idx_model_runs_status', table_name='model_runs')
    op.drop_index('idx_model_runs_experiment_id', table_name='model_runs')
    op.drop_index('idx_experiments_created_at', table_name='experiments')
    op.drop_index('idx_experiments_task_type', table_name='experiments')
    op.drop_index('idx_experiments_status', table_name='experiments')
    
    op.drop_table('performance_logs')
    op.drop_table('data_pipelines')
    op.drop_table('model_runs')
    op.drop_table('datasets')
    op.drop_table('experiments')
