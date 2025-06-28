import torch
import torch_neuronx
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import os

MODEL_ID_OR_S3_PATH = os.environ["MODEL_ID_OR_S3_PATH"]
MODEL_OUTPUT_S3_PATH = os.environ["MODEL_OUTPUT_S3_PATH"]

local_model_path="/tmp/raw-model"
save_directory="/tmp/neuron-model"
os.system(f"mkdir -p {save_directory}")

os.system(f"aws s3 cp --recursive --quiet {MODEL_ID_OR_S3_PATH} {local_model_path}")

SEQ_LEN = int(os.environ['SEQ_LEN'])
BATCH_SIZE = int(os.environ['BATCH_SIZE'])

tokenizer = AutoTokenizer.from_pretrained(local_model_path)

if os.environ['MODEL_TYPE'] == 'RERANK':
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
elif os.environ['MODEL_TYPE'] == 'EMB':
    model = AutoModel.from_pretrained(local_model_path)

model.eval()

texts = ['''Mr. Wilkins liked to feel his child dependent on him for all her pleasures.  He was even a little jealous of anyone who devised a treat or conferred a present, the first news of which did not come from or through him.
At last it was necessary that Ellinor should have some more instruction than her good old nurse could give.  Her father did not care to take upon himself the office of teacher, which he thought he foresaw would necessitate occasional blame, an occasional exercise of authority, which might possibly render him less idolized by his little girl; so he commissioned Lady Holster to choose out one among her many protegees for a governess to his daughter.  Now, Lady Holster, who kept a sort of amateur county register-office, was only too glad to be made of use in this way; but when she inquired a little further as to the sort of person required, all she could extract from Mr. Wilkins was:
"You know the kind of education a lady should have, and will, I am sure, choose a governess for Ellinor better than I could direct you. Only, please, choose some one who will not marry me, and who will let Ellinor go on making my tea, and doing pretty much what she likes, for she is so good they need not try to make her better, only to teach her what a lady should know."
Miss Monro was selected--a plain, intelligent, quiet woman of forty-- and it was difficult to decide whether she or Mr. Wilkins took the most pains to avoid each other, acting with regard to Ellinor, pretty much like the famous Adam:  when the one came out the other went in.  Miss Monro had been tossed about and overworked quite enough in her life not to value the privilege and indulgence of her evenings to herself, her comfortable schoolroom, her quiet cozy teas, her book, or her letter-writing afterwards.  By mutual agreement she did not interfere with Ellinor and her ways and occupations on the evenings when the girl had not her father for companion; and these occasions became more and more frequent as years passed on, and the deep shadow was lightened which the sudden death that had visited his household had cast over him.
'''] * BATCH_SIZE

encoded_input = tokenizer(
    texts,
    max_length=SEQ_LEN,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

example = (
    encoded_input['input_ids'],
    encoded_input['attention_mask'],
)

os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"

compiler_args = ["--optlevel=3",
            "--target=inf2",
            "--model-type=transformer", 
            "--auto-cast=matmult", 
            "--auto-cast-type=fp16"
            ]

model_neuron = torch_neuronx.trace(model, 
                example,
                compiler_args=compiler_args
                )

neuron_filename = f'{save_directory}/neuron-model.pt'

torch.jit.save(model_neuron, neuron_filename)
print(f"Saved model to {neuron_filename}")
tokenizer.save_pretrained(save_directory)
print(f"Saved tokenizer to {save_directory}")

os.system(f"aws s3 cp --recursive --quiet {save_directory} {MODEL_OUTPUT_S3_PATH}")

print(f"\n\nModel saved to {MODEL_OUTPUT_S3_PATH}\n\n")