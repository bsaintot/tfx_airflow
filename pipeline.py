import datetime
import os

import tensorflow_model_analysis as tfma
from tfx.components import CsvExampleGen, Evaluator, ExampleValidator, Pusher, ResolverNode, SchemaGen, StatisticsGen, \
    Trainer, Transform
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner, AirflowPipelineConfig
from tfx.proto import pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.utils.dsl_utils import external_input

pipeline_name = 'titanic'

airflow_dir = os.environ['AIRFLOW_HOME']
data_dir = os.path.join(airflow_dir, 'dags', 'data')

transform_module = os.path.join(airflow_dir, 'dags', 'transform.py')
train_module = os.path.join(airflow_dir, 'dags', 'trainer.py')

serving_model_dir = os.path.join(airflow_dir, 'serving_model', pipeline_name)

tfx_dir = os.path.join(airflow_dir, 'tfx')
pipeline_dir = os.path.join(tfx_dir, 'pipelines', pipeline_name)
metadata_path = os.path.join(tfx_dir, 'metadata', pipeline_name, 'metadata.db')

"""Implements the Titanic kaggle with TFX."""

example_gen = CsvExampleGen(input=external_input(data_dir))

statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'],
                       infer_feature_shape=False)

validate_stats = ExampleValidator(statistics=statistics_gen.outputs['statistics'],
                                  schema=schema_gen.outputs['schema'])

transform = Transform(examples=example_gen.outputs['examples'],
                      schema=schema_gen.outputs['schema'],
                      module_file=transform_module)

trainer = Trainer(module_file=train_module,
                  custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
                  examples=transform.outputs['transformed_examples'],
                  transform_graph=transform.outputs['transform_graph'],
                  schema=schema_gen.outputs['schema'],
                  train_args=trainer_pb2.TrainArgs(num_steps=10000),
                  eval_args=trainer_pb2.EvalArgs(num_steps=5000))

model_resolver = ResolverNode(instance_name='latest_blessed_model_resolver',
                              resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
                              model=Channel(type=Model),
                              model_blessing=Channel(type=ModelBlessing))

eval_config = tfma.EvalConfig(model_specs=[tfma.ModelSpec(label_key='Survived')],
                              slicing_specs=[tfma.SlicingSpec()],
                              metrics_specs=[tfma.MetricsSpec(thresholds={
                                  'binary_accuracy':
                                      tfma.config.MetricThreshold(
                                          value_threshold=tfma.GenericValueThreshold(
                                              lower_bound={'value': 0.6}),
                                          change_threshold=tfma.GenericChangeThreshold(
                                              direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                              absolute={'value': -1e-10}))})])

model_analyzer = Evaluator(examples=example_gen.outputs['examples'],
                           model=trainer.outputs['model'],
                           baseline_model=model_resolver.outputs['model'],
                           # Change threshold will be ignored if there is no baseline (first run).
                           eval_config=eval_config)

pusher = Pusher(model=trainer.outputs['model'],
                model_blessing=model_analyzer.outputs['blessing'],
                push_destination=pusher_pb2.PushDestination(
                    filesystem=pusher_pb2.PushDestination.Filesystem(
                        base_directory=serving_model_dir)))

tfx_pipeline = pipeline.Pipeline(pipeline_name=pipeline_name,
                                 pipeline_root=pipeline_dir,
                                 components=[example_gen, statistics_gen, schema_gen, validate_stats, transform,
                                             trainer, model_resolver, model_analyzer, pusher],
                                 enable_cache=True,
                                 metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
                                 # 0 means auto-detect based on on the number of CPUs available during
                                 # execution time.
                                 beam_pipeline_args=['--direct_num_workers=0'])

# 'DAG' below need to be kept for Airflow to detect dag.
airflow_config = {'schedule_interval': None, 'start_date': datetime.datetime(2019, 1, 1)}
DAG = AirflowDagRunner(AirflowPipelineConfig(airflow_config)).run(tfx_pipeline)
