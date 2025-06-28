import sagemaker

sess = sagemaker.Session()
role = 'arn:aws:iam::633205212955:role/service-role/AmazonSageMaker-ExecutionRole-20220923T160810'
# sagemaker_default_bucket = sess.default_bucket()
region = sess.boto_session.region_name

from sagemaker.estimator import Estimator

image_uri = f'763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training-neuronx:2.5.1-neuronx-py310-sdk2.22.0-ubuntu22.04'

model_tag = 'bge-reranker-base'
seqlen=512
bs=8

envs = {
    'MODEL_TYPE': 'RERANK' if 'reranker' in model_tag else 'EMB',
    'MODEL_ID_OR_S3_PATH': f's3://llm-artifacts-us-east-1/{model_tag}/',
    'MODEL_OUTPUT_S3_PATH': f"s3://zz-s3-us-e1/neuron-outputs-v1/compiled-output/",
    'SEQ_LEN': str(seqlen),
    'BATCH_SIZE': str(bs),
}

instance_type = "ml.trn1.2xlarge"

smp_estimator = Estimator(role=role,
    sagemaker_session=sess,
    base_job_name=f'neuron-trn1-compile',
    entry_point="model_trace.py",
    source_dir='submit_src',
    instance_type=instance_type,
    instance_count=1,
    environment=envs,
    # hyperparameters={},
    image_uri=image_uri,
    max_run=7200,
    keep_alive_period_in_seconds=3600,
    # enable_remote_debug=True,
    disable_output_compression=True,
)

smp_estimator.fit()








# s3://zz-s3-us-e1/neuron-outputs-rerank-v3/bge-reranker-base-seq512-bs1-0429-1206/
# s3://zz-s3-us-e1/neuron-outputs-rerank-v3/bge-reranker-base-seq512-bs2-0429-1206/
# s3://zz-s3-us-e1/neuron-outputs-rerank-v3/bge-reranker-base-seq512-bs4-0429-1206/
# s3://zz-s3-us-e1/neuron-outputs-rerank-v3/bge-reranker-base-seq512-bs8-0429-1206/
# s3://zz-s3-us-e1/neuron-outputs-rerank-v3/bge-reranker-base-seq512-bs12-0429-1206/
# s3://zz-s3-us-e1/neuron-outputs-rerank-v3/bge-reranker-base-seq512-bs16-0429-1206/
# s3://zz-s3-us-e1/neuron-outputs-rerank-v3/bge-reranker-base-seq512-bs20-0429-1206/
# s3://zz-s3-us-e1/neuron-outputs-rerank-v3/bge-reranker-base-seq512-bs24-0429-1206/

# s3://zz-s3-us-e1/neuron-outputs-emb-v1/neuron-bge-base-en-v1.5-seq512-bs1/
# s3://zz-s3-us-e1/neuron-outputs-emb-v1/neuron-bge-base-en-v1.5-seq512-bs2/
# s3://zz-s3-us-e1/neuron-outputs-emb-v1/neuron-bge-base-en-v1.5-seq512-bs4/
# s3://zz-s3-us-e1/neuron-outputs-emb-v1/neuron-bge-base-en-v1.5-seq512-bs8/
# s3://zz-s3-us-e1/neuron-outputs-emb-v1/neuron-bge-base-en-v1.5-seq512-bs12/
# s3://zz-s3-us-e1/neuron-outputs-emb-v1/neuron-bge-base-en-v1.5-seq512-bs16/
# s3://zz-s3-us-e1/neuron-outputs-emb-v1/neuron-bge-base-en-v1.5-seq512-bs20/
# s3://zz-s3-us-e1/neuron-outputs-emb-v1/neuron-bge-base-en-v1.5-seq512-bs24/

