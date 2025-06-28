import os
from datetime import datetime

TIMESTR = datetime.now().strftime("%m%d-%H%M")
from sagemaker.pytorch.model import PyTorchModel

role = f"arn:aws:iam::633205212955:role/service-role/AmazonSageMaker-ExecutionRole-20220923T160810"
image_uri = f"763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-neuronx:2.5.1-neuronx-py310-sdk2.22.0-ubuntu22.04"

save_directory = 'model_src'

model_s3_path = 's3://zz-s3-us-e1/neuron-outputs-v1/compiled-output/',

model_name = f'emb-seq512-bs8-dev'
max_len = 512

os.system(f"mkdir -p {save_directory}")
os.system(f"aws s3 cp {model_s3_path} {save_directory}/ --recursive --quiet")
os.system(f"mkdir -p {save_directory}/code")
os.system(f"cp -r inference.py {save_directory}/code/")

os.system(f"tar zcvf model.tar.gz -C {save_directory}/ .")
os.system(f"aws s3 cp model.tar.gz s3://zz-s3-us-e1/{save_directory}/{TIMESTR}/")

s3_model_uri = f"s3://zz-s3-us-e1/{save_directory}/{TIMESTR}/model.tar.gz"

pytorch_model = PyTorchModel(role=role,
                            name=model_name,
                            image_uri=image_uri,
                            model_data=s3_model_uri,
                            model_server_workers=2,
                            env={
                                "MAX_LENGTH": str(max_len),
                                "NEURON_RT_NUM_CORES": '1'
                            },
                        )

predictor = pytorch_model.deploy(instance_type='ml.inf2.xlarge',
                                initial_instance_count=1,
                                endpoint_name = model_name,
                                # wait=False,
                                volume_size = 200)

os.system(f"rm -rf model.tar.gz")
os.system(f"rm -rf {save_directory}")