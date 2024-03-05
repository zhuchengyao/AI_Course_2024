# This is Fudan AI Course by Patrick Jiang


# Huggingface introduction

# Models
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset



def conversation(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. "
                                      "You are going to answer some multiple choices questions"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


# dataset loading
cais_datasets = load_dataset("cais/mmlu", "all", trust_remote_code=True)
case = cais_datasets["auxiliary_train"][0]
# model loading
device = "cpu" # the device to load the model onto
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-0.5B-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")



for i in range(3):
    case = cais_datasets["auxiliary_train"][i]
    prompt = case["question"]
    for num, answer in enumerate(case["choices"]):
        prompt += prompt + ' ' + str(num+1) + ': ' + answer + '. '
    print(prompt)
    conversation(prompt)
    print(case['answer'])









