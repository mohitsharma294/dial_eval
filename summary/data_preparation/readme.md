Run below script to download 4 datasets: (CNN_DAILYMAIL, XSUM, Wikisum, Webis TLDR). It will create a folder summarisation_data.

python3 download_summary_data.py


Once the all the dataset is downloaded, run below script to extract the data from different dataset and combine them into one.

Ex1: to get 0.1 million data from CNN_DAILYMAIL and 0.2 million data from Wikisum run the following script

python3 extract_data.py -cnn 0.1 -wikisum 0.3 -filetype csv

Ex2: to fetch 0.1 million data from each of the dataset and save into json file with specifying outout file name run:

python3 extract_data.py -cnn 0.1 -webis 0.1 -xsum 0.1 -wikisum 0.1 -filetype json -filename output_file

Note: supported filetype are : json, parquet and csv. for each flag of dataset mention the size in million.


