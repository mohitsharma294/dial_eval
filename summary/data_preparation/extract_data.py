import os
import glob
import pandas as pd

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-cnn", "--cnn", type=float, default = 0,
                    help="cnn records in million")
parser.add_argument("-wikisum", "--wikisum",type=float, default = 0,
                    help="wikisum records in million")
parser.add_argument("-webis", "--webis",type=float, default = 0,
                    help="webis records in million")
parser.add_argument("-xsum", "--xsum",type=float, default = 0,
                    help="xsum records in million")
parser.add_argument("-filetype", "--filetype", default = "json",
                    help="output file type")

parser.add_argument("-filename", "--filename", default = "summarization_data",
                    help="output file name")


args = parser.parse_args()

cnn = args.cnn
wikisum = args.wikisum
webis = args.webis
xsum = args.xsum
filetype = args.filetype
filename = args.filename

cnn_records = cnn*1000000
wikisum_records = wikisum*1000000
webis_records = webis * 1000000
xsum_records = xsum * 1000000


download_folder = "summarisation_data"
cnn_dailymail_folder = os.path.join(download_folder, "cnn_dailymail")
wikisum_folder = os.path.join(download_folder, "wiki_sum")
wikissum_parquet_chunks = os.path.join(wikisum_folder, "parquet_chunks")
webis_tldr_folder = os.path.join(download_folder, "webis_tldr")
xsum_folder = os.path.join(download_folder, "xsum")


cnn_df = pd.DataFrame(columns = ["document", "summary", "dataset"])
if cnn_records:
    cnn_files = glob.glob(f"{cnn_dailymail_folder}/*.parquet")
    cnn_len = 0 
    for i in cnn_files:
        curr_df = pd.read_parquet(i)
        curr_df = curr_df.rename(columns = {"article":"document", "highlights":"summary"})[["document", "summary"]]
        cnn_df = pd.concat([cnn_df, curr_df])
        if len(cnn_df) >= cnn_records:
            break
    cnn_df = cnn_df.iloc[:int(cnn_records), :]
    cnn_df['dataset'] = "cnn_dailymail"
    
    
wiksum_df = pd.DataFrame(columns = ["document", "summary", "dataset"])
if wikisum_records:
    wikisum_files = glob.glob(f"{wikissum_parquet_chunks}/*.parquet")
    wiksum_len = 0 
    for i in wikisum_files:
        curr_df = pd.read_parquet(i)
        curr_df = curr_df.rename(columns = {"full_text":"document"})[["document", "summary"]]
        wiksum_df = pd.concat([wiksum_df, curr_df])
        if len(wiksum_df) >= wikisum_records:
            break
    wiksum_df = wiksum_df.iloc[:int(wikisum_records), :]
    wiksum_df['dataset'] = "wikisum"
    
webis_df = pd.DataFrame(columns = ["document", "summary", "dataset"])
if webis_records:
    webis_files = glob.glob(f"{webis_tldr_folder}/*.parquet")
    webis_len = 0 
    for i in webis_files:
        curr_df = pd.read_parquet(i)
        curr_df = curr_df.rename(columns = {"content":"document"})[["document", "summary"]]
        webis_df = pd.concat([webis_df, curr_df])
        if len(webis_df) >= webis_records:
            break
    webis_df = webis_df.iloc[:int(webis_records), :]
    webis_df['dataset'] = "webis_tldr"

xsum_df = pd.DataFrame(columns = ["document", "summary", "dataset"])
if xsum_records:
    xsum_files = glob.glob(f"{xsum_folder}/*.parquet")
    xsum_len = 0 
    for i in xsum_files:
        curr_df = pd.read_parquet(i)
        xsum_df = pd.concat([xsum_df, curr_df])
        if len(webis_df) >= xsum_records:
            break
    xsum_df = xsum_df.iloc[:int(xsum_records), :]
    xsum_df['dataset'] = "xsum"
    
#combine all df
all_df = pd.concat([cnn_df, wiksum_df, webis_df, xsum_df])
if filetype == "json":
    all_df.to_json(f"{filename}.json", orient = "records",index = "false")
elif filetype == "csv":
    all_df.to_csv(f"{filename}.csv", index = False)
elif filetype == "parquet":
    all_df.to_parquet(f"{filename}.parquet")
