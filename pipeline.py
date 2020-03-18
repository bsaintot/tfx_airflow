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

_pipeline_name = 'taxi_chicago'

_taxi_root = os.path.join(os.environ['HOME'], 'airflow')
_data_root = os.path.join(_taxi_root, 'data', 'taxi_data')

# Transform and Trainer both require user-defined functions to run successfully.
_taxi_transform_module_file = os.path.join(_taxi_root, 'dags', 'transform.py')
_taxi_trainer_module_file = os.path.join(_taxi_root, 'dags', 'trainer.py')

# Path which can be listened to by the model server.  Pusher will output the trained model here.
_serving_model_dir = os.path.join(_taxi_root, 'serving_model', _pipeline_name)


_tfx_root = os.path.join(_taxi_root, 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name, 'metadata.db')

# Airflow-specific configs; these will be passed directly to airflow
_airflow_config = {'schedule_interval': None, 'start_date': datetime.datetime(2019, 1, 1)}


"""Implements the chicago taxi pipeline with TFX."""

# Brings data into the pipeline or otherwise joins/converts training data.
example_gen = CsvExampleGen(input=external_input(_data_root))

# Computes statistics over data for visualization and example validation.
statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

# Generates schema based on statistics files.
infer_schema = SchemaGen(statistics=statistics_gen.outputs['statistics'],
                         infer_feature_shape=False)

# Performs anomaly detection based on statistics and data schema.
validate_stats = ExampleValidator(statistics=statistics_gen.outputs['statistics'],
                                  schema=infer_schema.outputs['schema'])

# Performs transformations and feature engineering in training and serving.
transform = Transform(examples=example_gen.outputs['examples'],
                      schema=infer_schema.outputs['schema'],
                      module_file=_taxi_transform_module_file)

# Uses user-provided Python function that implements a model using TF-Learn.
trainer = Trainer(module_file=_taxi_trainer_module_file,
                  custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
                  examples=transform.outputs['transformed_examples'],
                  transform_graph=transform.outputs['transform_graph'],
                  schema=infer_schema.outputs['schema'],
                  train_args=trainer_pb2.TrainArgs(num_steps=10000),
                  eval_args=trainer_pb2.EvalArgs(num_steps=5000))

# Get the latest blessed model for model validation.
model_resolver = ResolverNode(instance_name='latest_blessed_model_resolver',
                              resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
                              model=Channel(type=Model),
                              model_blessing=Channel(type=ModelBlessing))

# Uses TFMA to compute a evaluation statistics over features of a model and
# perform quality validation of a candidate model (compared to a baseline).
eval_config = tfma.EvalConfig(model_specs=[tfma.ModelSpec(label_key='tips')],
                              slicing_specs=[tfma.SlicingSpec()],
                              metrics_specs=[
                                  tfma.MetricsSpec(
                                      thresholds={
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

# Checks whether the model passed the validation steps and pushes the model to a file destination if check passed.
pusher = Pusher(model=trainer.outputs['model'],
                model_blessing=model_analyzer.outputs['blessing'],
                push_destination=pusher_pb2.PushDestination(
                    filesystem=pusher_pb2.PushDestination.Filesystem(
                        base_directory=_serving_model_dir)))

tfx_pipeline = pipeline.Pipeline(pipeline_name=_pipeline_name,
                                 pipeline_root=_pipeline_root,
                                 components=[example_gen, statistics_gen, infer_schema, validate_stats, transform,
                                             trainer, model_resolver, model_analyzer, pusher],
                                 enable_cache=True,
                                 metadata_connection_config=metadata.sqlite_metadata_connection_config(_metadata_path),
                                 # TODO(b/142684737): The multi-processing API might change.
                                 # 0 means auto-detect based on on the number of CPUs available during
                                 # execution time.
                                 beam_pipeline_args=['--direct_num_workers=0'])

# 'DAG' below need to be kept for Airflow to detect dag.
DAG = AirflowDagRunner(AirflowPipelineConfig(_airflow_config)).run(tfx_pipeline)
