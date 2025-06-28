
import os, time, json
import torch
# import torch_neuronx
from transformers import AutoTokenizer, AutoModel

MAX_LENGTH = int(os.environ['MAX_LENGTH'])
print("Set MAX_LENGTH from ENV: ", MAX_LENGTH)


def model_fn(model_dir, context):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = torch.jit.load(model_dir+"/neuron-model.pt")
    print("### model_fn loaded ###")
    return model, tokenizer


def input_fn(request_body, request_content_type):
    input_data = json.loads(request_body)
    return input_data


def predict_fn(data, pipeline):
    model, tokenizer = pipeline

    inputs_data = data.pop("inputs", "ERROR")

    st = time.perf_counter()

    encoded_input = tokenizer(
        inputs_data,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    example = (
        encoded_input['input_ids'],
        encoded_input['attention_mask'],
    )

    output_neuron = model(*example)

    t3 = time.perf_counter() - st

    return {"result": output_neuron['pooler_output'].tolist(), "total_inf_time": t3}

def output_fn(prediction, content_type):
    return json.dumps(prediction)
