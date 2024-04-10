import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# load model
model_checkpoint = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                           trust_remote_code=True,
                                           padding_side="left",)
device = "cuda:4"
batch_size  = 10
max_new_tokens = 50

model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    pad_token_id=tokenizer.eos_token_id,
    device_map="auto", 
    max_memory = {4:"40GiB"},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

# load data
train_file = '/local/xiaowang/food_ingredient/Dataset_440_labels/1mtest_train_440_labels.json'
test_file = '/local/xiaowang/food_ingredient/Dataset_440_labels/1mtest_test_440_labels.json'
val_file = '/local/xiaowang/food_ingredient/Dataset_440_labels/1mtest_val_440_labels.json'

train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
val_df = pd.read_json(val_file)

train_titles = train_df['title'].tolist() # (48388, 10)
test_titles = test_df['title'].tolist() # (6048, 10) 
val_titles = val_df['title'].tolist() #  (6049, 10)

# create messages using chat templates 
message = [
    {"role": "user", "content": 'Breifly introduce the recipe:{}'},
]

train_messages = [[{"role": "user", "content": f'Briefly introduce the recipe: {title}'}] for title in train_titles]
test_messages = [[{"role": "user", "content": f'Briefly introduce the recipe: {title}'}] for title in test_titles]
val_messages = [[{"role": "user", "content": f'Briefly introduce the recipe: {title}'}] for title in val_titles]

# element in list: '<s>[INST] Briefly introduce the recipe: Anzac Biscuits With Macadamias (Australian) [/INST]'
train_input_txt_ls = [tokenizer.apply_chat_template(train_message, tokenize=False, add_generation_prompt=True, return_dict=False, return_tensors="pt") for train_message in train_messages]
test_input_txt_ls = [tokenizer.apply_chat_template(test_message, tokenize=False, add_generation_prompt=True, return_dict=False, return_tensors="pt") for test_message in test_messages]
val_input_txt_ls = [tokenizer.apply_chat_template(val_message, tokenize=False, add_generation_prompt=True, return_dict=False, return_tensors="pt") for val_message in val_messages]

#! generate instructions
tokenizer.pad_token = tokenizer.eos_token


# training set

iteration_idx = 1
train_extracted_outputs = []

print('Generating instructions for training set')

for iteration_idx in tqdm(range(1, (len(train_input_txt_ls) + batch_size - 1) // batch_size + 1)):
    batch_input_txt_ls = train_input_txt_ls[(iteration_idx-1)*batch_size : iteration_idx*batch_size]
    batch_input_ids = tokenizer(batch_input_txt_ls, return_tensors="pt", padding=True, truncation=True).to(device)
    message_lens = batch_input_ids['attention_mask'][0].nelement() # to extract model generated tokens
    generated_ids = model.generate(**batch_input_ids, max_new_tokens=max_new_tokens, do_sample=True)
    extracted_outputs = tokenizer.batch_decode(generated_ids[:, message_lens:], skip_special_tokens=True)
    train_extracted_outputs.extend(extracted_outputs)
    

train_extracted_outputs_df = pd.DataFrame(train_extracted_outputs)
train_extracted_outputs_df.to_csv('train_generated_instructions.csv', index=False)
print('Instructions generated successfully!')

# test set

iteration_idx = 1
test_extracted_outputs = []

print('Generating instructions for test set')

for iteration_idx in tqdm(range(1, (len(test_input_txt_ls) + batch_size - 1) // batch_size + 1)):
    batch_input_txt_ls = test_input_txt_ls[(iteration_idx-1)*batch_size : iteration_idx*batch_size]
    batch_input_ids = tokenizer(batch_input_txt_ls, return_tensors="pt", padding=True, truncation=True).to(device)
    message_lens = batch_input_ids['attention_mask'][0].nelement() # to extract model generated tokens
    generated_ids = model.generate(**batch_input_ids, max_new_tokens=max_new_tokens, do_sample=True)
    extracted_outputs = tokenizer.batch_decode(generated_ids[:, message_lens:], skip_special_tokens=True)
    test_extracted_outputs.extend(extracted_outputs)

test_extracted_outputs_df = pd.DataFrame(test_extracted_outputs)
test_extracted_outputs_df.to_csv('test_generated_instructions.csv', index=False)
print('Instructions generated successfully!')

# validation set

iteration_idx = 1
val_extracted_outputs = []

print('Generating instructions for validation set')

for iteration_idx in tqdm(range(1, (len(val_input_txt_ls) + batch_size - 1) // batch_size + 1)):
    batch_input_txt_ls = val_input_txt_ls[(iteration_idx-1)*batch_size : iteration_idx*batch_size]
    batch_input_ids = tokenizer(batch_input_txt_ls, return_tensors="pt", padding=True, truncation=True).to(device)
    message_lens = batch_input_ids['attention_mask'][0].nelement() # to extract model generated tokens
    generated_ids = model.generate(**batch_input_ids, max_new_tokens=max_new_tokens, do_sample=True)
    extracted_outputs = tokenizer.batch_decode(generated_ids[:, message_lens:], skip_special_tokens=True)
    val_extracted_outputs.extend(extracted_outputs)

val_extracted_outputs_df = pd.DataFrame(val_extracted_outputs)
val_extracted_outputs_df.to_csv('val_generated_instructions.csv', index=False)
print('Instructions generated successfully!')
