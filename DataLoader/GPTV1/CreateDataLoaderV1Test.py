from CreateDataLoaderV1 import create_data_loader

with open("LLM/Data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_data_loader(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iterator = iter(dataloader)
first_batch = next(data_iterator)
second_batch = next(data_iterator)


print('First Batch : \n ' , first_batch) 
print('\nSecond Batch : \n ' , second_batch) 
