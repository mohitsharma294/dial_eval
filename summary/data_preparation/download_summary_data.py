import os
import urllib.request
import traceback
import glob
from pyarrow.parquet import ParquetFile
import pyarrow as pa 
import pandas as pd

download_folder = "summarisation_data"
cnn_dailymail_folder = os.path.join(download_folder, "cnn_dailymail")
wikisum_folder = os.path.join(download_folder, "wiki_sum")
wikissum_parquet_chunks = os.path.join(wikisum_folder, "parquet_chunks")
webis_tldr_folder = os.path.join(download_folder, "webis_tldr")
xsum_folder = os.path.join(download_folder, "xsum")
for folder in [download_folder, cnn_dailymail_folder, wikisum_folder, wikissum_parquet_chunks, webis_tldr_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        
        
#########################################
#                                       #
#          CNN DAILYMAIL                #
#                                       #
#########################################
print("Downloading cnn daily mail summarisation data, if not exist...")
cnn_dailymail_url_folder = "https://huggingface.co/datasets/cnn_dailymail/resolve/refs%2Fconvert%2Fparquet/3.0.0/train/"

for i in range(3):
    url = cnn_dailymail_url_folder + "{:04d}.parquet".format(i)
    file_name = os.path.join(cnn_dailymail_folder, url.split("/")[-1])
    if os.path.isfile(file_name):
        print(f"{file_name} file already exist, skipping download")
    else:
        print(f"Downlaoding file: {file_name}")
        try:
            print(url)
            os.system(f"wget {url} -O {file_name}")
        except Exception as e:
            print(f"Exception occured while downloading file : {str(e)}")
            traceback.print_exc()
            
#########################################
#                                       #
#          WIKI SUM DATA                #
#                                       #
#########################################            
print("Downloading wikipedia summarisation data, if not exist...")
wiki_sum_url = "https://huggingface.co/datasets/jordiclive/wikipedia-summary-dataset/resolve/main/df_withDescription.parquet"
file_name = os.path.join(wikisum_folder, wiki_sum_url.split("/")[-1])
if os.path.isfile(file_name):
    print(f"{file_name} file already exist, skipping download")
else:
    print(f"Downlaoding file: {file_name}")
    try:
        print(wiki_sum_url)
        os.system(f"wget {wiki_sum_url} -O {file_name}")
    except Exception as e:
        print(f"Exception occured while downloading file : {str(e)}")
        traceback.print_exc()
# check if file is already divided into chunks or not?
if len(glob.glob(f"{wikissum_parquet_chunks}/*.parquet")) < 47:
    print("Dividing wikisum file into multiple parquet files")
    print("total batches to be built 47")
    pf = ParquetFile(file_name) 
    c = 0
    for batch in pf.iter_batches():
        print(c, end = "\r")
        parquet_name = os.path.join(wikissum_parquet_chunks, "{:04d}.parquet".format(c))
        if not os.path.isfile(parquet_name):
            batch_df = batch.to_pandas()
            batch_df.to_parquet(parquet_name)
        c+=1
            
#########################################
#                                       #
#           WEBIS TLDR                  #
#                                       #
#########################################
print("Downloading WEBIS TLDR summarisation data, if not exist...")
webis_parent_url = "https://huggingface.co/datasets/webis/tldr-17/resolve/refs%2Fconvert%2Fparquet/default/partial-train/"

for i in range(10):
    url = webis_parent_url + "{:04d}.parquet".format(i)
    file_name = os.path.join(webis_tldr_folder, url.split("/")[-1])
    if os.path.isfile(file_name):
        print(f"{file_name} file already exist, skipping download")
    else:
        print(f"Downlaoding file: {file_name}")
        try:
            print(url)
            os.system(f"wget {url} -O {file_name}")
        except Exception as e:
            print(f"Exception occured while downloading file : {str(e)}")
            traceback.print_exc()
        
#########################################
#                                       #
#           XSUM DATA                   #
#                                       #
#########################################    

print("Downloading XSUM summarisation data, if not exist...")
xsum_download_url = "https://huggingface.co/datasets/xsum/resolve/main/data/XSUM-EMNLP18-Summary-Data-Original.tar.gz"
file_name = os.path.join(xsum_folder, xsum_download_url.split("/")[-1])
if os.path.isfile(file_name):
    print(f"{file_name} file already exist, skipping download")
else:
    os.system(f"wget {xsum_download_url} -P {xsum_folder}")
extracted_folder = os.path.join(xsum_folder, "bbc-summary-data")
if not os.path.exists(extracted_folder):
    print("Extracting file")
    os.system(f"tar -xzf {file_name} -C {xsum_folder}")
parquet_file = os.path.join(xsum_folder, "0000.parquet")
print("combining the all summary file to single parquet file")
# reading and changing to parquet file
if not os.path.isfile(parquet_file):
    REMOVE_LINES = set(
        [
            "Share this with\n",
            "Email\n",
            "Facebook\n",
            "Messenger\n",
            "Twitter\n",
            "Pinterest\n",
            "WhatsApp\n",
            "Linkedin\n",
            "LinkedIn\n",
            "Copy this link\n",
            "These are external links and will open in a new window\n",
        ]
    )

    all_summary_files = glob.glob(f"{extracted_folder}/*.summary")
    print(f"reading all files one by one, total files: {len(all_summary_files)}")
    c = 0
    document = []
    summary = []
    for file in all_summary_files:
        print(c, end = "\r")
        c+=1
        try:
            with open(file, "r") as f:
                text = "".join([line for line in f.readlines() 
                                if line not in REMOVE_LINES and line.strip()])
                segments= text.split("[SN]")
                doc_ = segments[8].strip()
                summary_ =  segments[6].strip()
                document.append(doc_)
                summary.append(summary_)
        except Exception as exp:
            pass
    df = pd.DataFrame({"document":document, "summary":summary})
    df.to_parquet(parquet_file)