from CreateDataLoaderV1 import CreateDataLoaderV1

with open("LLM/Data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = CreateDataLoaderV1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iterator = iter(dataloader)
first_batch = next(data_iterator)
second_batch = next(data_iterator)


print('First Batch : ' , first_batch) 
print('Second Batch : ' , second_batch) 
