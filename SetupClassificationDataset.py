import urllib.request
import zipfile
import os
import pandas as pd
from pathlib import Path



url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(
        url, zip_path,extracted_path, data_file_path ):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. skipping download and extraction")
        return
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
        
    with zipfile.ZipFile(zip_path, "r") as zip_ref:    
            zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File Downloaded and saved as  {data_file_path}")



def create_balanced_dataset(dataframe):
     num_spam = dataframe[dataframe["Label"] == "spam"].shape[0]
     ham_subset = dataframe[dataframe["Label"] == "ham"].sample(
          num_spam, random_state=123
     )
     balanced_dataframe = pd.concat([
          ham_subset, dataframe[dataframe["Label"] == "spam"]
     ])
     return balanced_dataframe

def random_split(dataframe, train_frac, validation_frac):
     dataframe = dataframe.sample(
          frac=1, random_state=123
     ).reset_index(drop=True)
     train_end = int(len(dataframe) * train_frac)
     validation_end = train_end + int(len(dataframe) * validation_frac)

     train_dataframe = dataframe[:train_end]
     validation_dataframe = dataframe[train_end:validation_end]
     test_dataframe = dataframe[validation_end:]

     return train_dataframe, validation_dataframe, test_dataframe



download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)


dataframe = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["Label","Text"]
)


balanced_dataframe = create_balanced_dataset(dataframe)
print(balanced_dataframe["Label"].value_counts())

balanced_dataframe["Label"] = balanced_dataframe["Label"].map({"ham":0,"spam":1})

train_dataframe, validation_dataframe, test_dataframe = random_split(dataframe, .7, .1)


train_dataframe.to_csv("train.csv", index=None)
validation_dataframe.to_csv("validation.csv", index=None)
test_dataframe.to_csv("test.csv", index=None)