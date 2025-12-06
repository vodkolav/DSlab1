
import json
from urllib.parse import urlparse, parse_qs
#import urllib3
import urllib

import re
import requests
import time
import os
import pandas as pd
import kagglehub

import zenodo_get

def jload(fn = "secrets.json"):
    with open(fn, 'r') as f:
        return json.load(f)


def jsave(tree, fn):
    with open(fn, 'w+') as f:
        return json.dump(tree, f, indent=2)
    

secrets = jload("secrets.json")

# def md_save(df, fn):
#     mcw = [10] * 5 + [None] + [40,40]
#     with open(fn,'w+') as f:
#         df.to_markdown(f, index = False, tablefmt="pipe", maxcolwidths=mcw)



# def md_save(df, folder, tblname):
#     os.makedirs(folder + "/img", exist_ok=True)
#     tblname = os.path.splitext(tblname)[0]
#     # save images
#     for i,r in df.iterrows():

#         imgname = f'img/{tblname}_{r.Column}.png'
#         imgname = imgname.replace(" ","_").replace(":","_")

#         with open(folder + "/" + imgname, 'wb') as f:
#             f.write(r.Hist)
#         r.Hist = f'![Histogram]({imgname})'

#     #mcw = [10] * 5 + [None] + [40,40]
    
#     md = df.to_markdown(index = False, tablefmt="pipe") # , maxcolwidths=mcw)
#     md = re.sub(r'-{42,}', '-'*42, md)  # shorten columns width
#     with open(folder + "/" + tblname + ".md" ,'w+') as f:
#         f.write(md)



def to_md(df, itemname, tblname):
    
    tblname = os.path.splitext(tblname)[0] # only get name, without extension
    # prepare images links

    assets = {}
    for i,r in df.iterrows():
        
        
        imgname = f'{itemname}_{tblname}_{r.Column}.png'
        imgname = imgname.replace(" ","_").replace(":","_")

        assets[imgname] = r.Hist
        r.Hist = f'![Histogram](img/{imgname})'
    
    md = df.to_markdown(index = False, tablefmt="pipe") # , maxcolwidths=mcw)
    md = re.sub(r'-{42,}', '-'*42, md)  # shorten columns width
    return md, assets


def md_save(outputs_root, itemname, md, assets):

    os.makedirs(os.path.join(outputs_root, "img"), exist_ok=True)

    for imgname, data in assets.items():
        with open( os.path.join(outputs_root ,"img", imgname), 'wb') as f:
            f.write(data)

    with open(os.path.join(outputs_root, itemname + ".md") ,'w+') as f:
        f.write(md)




def read_yaml(file):
    # read_yaml(pth) runs too long. why?
    import yaml
    with open(file, 'r') as f:
        return pd.json_normalize(yaml.safe_load(f))
    


import warnings

def read_any(file):
    if file.endswith(('.csv', 'tsv')) :
        df = pd.read_csv(file, sep=None, engine='python')
    elif file.endswith('.json'):
        df = pd.read_json(file)
    elif file.endswith('.jsonl'):
        df = pd.read_json(file, lines=True)
    elif file.endswith('.yaml'):
        df = read_yaml(file)
    elif file.endswith('.xml'):
        df = pd.read_xml(file)
    elif file.endswith(('.xls','xlsx')):
        df = pd.read_excel(file)
    elif file.endswith('.hdf'):
        df = pd.read_hdf(file)           
    elif file.endswith('.sql'):
        df = pd.read_sql(file)
    else:        
        warnings.warn(f'Skipping Unsupported filetype: {file}')
        df = None
    return df




def download_webpage(url, filename ):
    if os.path.exists(filename):
        print(f"skipping download - already exists: {filename}.")
        return 'skip'
        
    time.sleep(.5)  # Sleep for 0.5 second to avoid overwhelming the server
    response = requests.get(url)

    if response.status_code == 200:  # Check if the page was successfully fetched
        with open(filename, "w") as f:
            f.write(response.text)
        print(f"Downloaded {url} to {filename}")
        return 'ok'
    
    else:        
        newurl = url.replace("_all","")
        if newurl != url:
            print(f"Failed to retrieve {url}, trying alternative")
            return download_webpage(newurl, filename)
        else:
            print(f"Failed to retrieve {url} and no alternative available.")
            return 'fail'        


def EXPLOSON(df, which_col = 'labels'):
    # specific to EduRABSA_Dataset
    #df = pd.read_json(fl, lines=True)
    df = df.explode(which_col)\
        .reset_index(drop=False, names = ["review_id"])
    tp = pd.json_normalize(df[which_col])
    df = pd.merge(df, tp, left_index=True, right_index=True)\
        .drop(which_col, axis=1)
    return df

def collect_subtable(base_url, loc_path, tbl):
    df = []
    tbl = pd.DataFrame(tbl[0].to_list(), columns=['Term','Path'])
    tbl.dropna(axis=0, inplace=True)
    # tbl.columns = [trm[0] for trm in tbl.iloc[-2]]
    # tbl.drop(tbl.index[-2:], inplace=True)

    for i,r in tbl.iterrows():
        end = r["Path"]
        suburl = base_url + urllib.parse.quote( end, safe=':/&', encoding=None, errors=None) 
        dest = os.path.join(loc_path , end)
        dfname = os.path.dirname(end)
        os.makedirs(os.path.join(loc_path, dfname ), exist_ok=True)

        status = download_webpage(suburl, dest)
        if status != 'fail':
            df.append(pd.read_html(dest)[0]) # , extract_links="body"

    df = pd.concat(df)
    return df, dfname

#df = get_suburl(base_url, welcome)





def download(item):
    # Parse the URL into its components
    parsed_url = urlparse(item.url)
    
    # Access individual components
    scheme = parsed_url.scheme
    domain = parsed_url.hostname
    netloc = parsed_url.netloc
    path = parsed_url.path
    query = parsed_url.query
    fragment = parsed_url.fragment

    
    print("downloading", item.url)


    dfs = {}
    match item.repository.lower() :
        case "kaggle":
            ds_path = path.replace("/datasets/", "")
            loc_path = kagglehub.dataset_download(ds_path)
            print("downloaded to", loc_path)

            dfs = load_dir(loc_path)
                    
        case "huggingface":
            ds_path = path.replace("/datasets/", "")            
            builder = load_dataset_builder(ds_path)
            builder.download_and_prepare()
            loc_path = builder.cache_dir
            print("downloaded to", loc_path)
            files = os.listdir(loc_path)
            print("files:", files)
            dfs = builder.as_dataset()
            dfs = {k: v.to_pandas() for k,v in dfs.items()}
            #files = list(dfs.keys())
            #dfs[0] = ds["train"]

        case "lis-lab.fr":

            # TODO: download all htmls from page
            loc_path = os.path.expanduser(f"~/.cache/{item.repository.lower()}/{item.citekey}")
            base_url = item.url
            os.makedirs(loc_path, exist_ok=True)
            filename = urllib.parse.urlparse(base_url.strip('/'))\
                            .path.split(os.path.sep)[-1] + ".html"
            dest = os.path.join(loc_path , filename)
            status = download_webpage(base_url, dest)

            att = {"class":"tableau zebre"}
            summ_tables = pd.read_html(dest,extract_links="all", attrs= att)

            dfs = {}
            for st in summ_tables:
                try :
                    df,nm = collect_subtable(base_url, loc_path, st)
                    dfs[nm] = df
                except Exception as e:
                    print("failed to collect subtable", e)
                    continue

            # # TODO: download all htmls from page
            
            # url = "https://pageperso.lis-lab.fr/ismail.badache/Reviews_ExtracTerms%20HTML/Aspect%20Terms%20Sentiment%20M1%20NLTK/Reviews_ExtracTerm_C_Sent_Assignment.html"
            # os.makedirs(loc_path, exist_ok=True)
            # filename = "Reviews_ExtracTerm_C_Sent_Assignment.html"
            # dest = os.path.join(loc_path , filename)
            # status = download_webpage(url, dest)
            # if status != 'fail':
            #     dfs[filename] = pd.read_html(dest)[0]

        case "zenodo":
            ds_path = path.replace("/records/", "")            
            loc_path = os.path.expanduser(f"~/.cache/{item.repository.lower()}/{item.citekey}")
            os.makedirs(loc_path, exist_ok=True)
            zenodo_get.download( ds_path, output_dir=loc_path ) #,   unzip=True) #, verbose=True)

            
            #loc_path = loc_path + "/EduRABSA_Dataset/2_annotated_dataset_files/pyabsa_dataset_format/ASQE/"
            dfs = load_dir(loc_path)
                #dfs = {k: transform(v) for k,v in dfs.items()}

        case "hesa":
            loc_path = os.path.expanduser(f"~/.cache/{item.repository.lower()}/{item.citekey}")
            os.makedirs(loc_path, exist_ok=True)
            files = os.listdir(loc_path)
            if files:
                dfs = load_dir(loc_path)
            else:
                print("no files in", loc_path, "download them manually and put there")
        
        case "manual":
            loc_path = os.path.expanduser(f"~/.cache/{item.repository.lower()}/{item.citekey}")
            os.makedirs(loc_path, exist_ok=True)
            files = os.listdir(loc_path)
            if files:
                dfs = load_dir(loc_path)
            else:
                print("no files in", loc_path, "download them manually and put there")
        

        case _ :
            print("skipping not implemented", item.citekey, "from", item.repository )

    if item.citekey in need_preprocess:
        transform = need_preprocess[item.citekey]
        dfs = {k: transform(v) for k,v in dfs.items()}

    return dfs


def load_dir(loc_path):
    dfs = {}
    files = os.listdir(loc_path)
    print("files:", files)
    for f in files:
        pth = os.path.join(loc_path, f)
        df = read_any(pth)
        if df is not None:
            dfs[f] = df
    return dfs
