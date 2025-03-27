from unsloth import FastLanguageModel
import json
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from utils.format import create_mol_format
from utils.config import AtomTypeEnumZinc, BondTypeEnumZinc, AtomTypeEnumQM9, BondTypeEnumQM9, AtomTypeEnum, BondTypeEnum
from utils.tree2smiles import tree2smiles
import os
from tqdm import tqdm

model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "saved_models/Qwen2_5_3B_Instruct_lora_model", # YOUR MODEL YOU USED FOR TRAINING
    model_name = "saved_models/Qwen2_5_3B_Instruct_lora_model_sft_5",
    max_seq_length = 3500,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

mol_schema, _ = create_mol_format(AtomTypeEnumZinc, BondTypeEnumZinc)

class_schema = json.dumps(mol_schema.model_json_schema(), indent=0)
parser = JsonSchemaParser(mol_schema.model_json_schema())
prefix_function = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

inputs = tokenizer(
[
    alpaca_prompt.format(
        f"Please generate a valid molecule from the following starting structure. You must respond using JSON format, according to the following schema: {class_schema}.", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer)

# Generate a molecule
num_samples = 1000
json_data_list = []
invalid_json = []
valid_molecules = []
for i in tqdm(range(num_samples)):
    generated_ids = model.generate(**inputs, prefix_allowed_tokens_fn=prefix_function, max_new_tokens = 4000, do_sample=True, top_k=50, top_p=0.95, temperature=0.8, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    prompt = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    # print(prompt)
    # print(output[0])
    # print("=============================================================")
    # print(output[0][len(prompt):])
    try:
        molecule_data = json.loads(output[0][len(prompt):])
        json_data_list.append(molecule_data)
        path = 'results/json/qwen_zinc_sft_5/'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path+str(i)+'.json', 'w') as file:
            json.dump(molecule_data, file, indent=0)
    except Exception as e:
        print(e)
        print('Invalid JSON')
        invalid_json.append(output[0][len(prompt):])
        continue
    smiles = tree2smiles(molecule_data, do_correct=False)
    if smiles:
        valid_molecules.append(smiles)
        path = 'results/smiles/qwen_zinc_sft_5.txt'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'a') as f:
            f.write(smiles + '\n')