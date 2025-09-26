import os
import re
import requests
import numpy as np
from urllib.parse import urljoin
from netCDF4 import Dataset
from chromadb import CloudClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://data-argo.ifremer.fr/dac/aoml/"

def list_links(url, pattern=None):
    """Return all href links from a directory URL, optionally filtered by regex."""
    resp = requests.get(url)
    resp.raise_for_status()
    links = re.findall(r'href="([^"]+)"', resp.text)
    if pattern:
        links = [l for l in links if re.search(pattern, l)]
    return links

def recursive_nc_files(base_url):
    """Crawl float directories and yield full URLs of .nc files."""
    float_dirs = list_links(base_url, r'^[0-9]+/$')   # e.g. 1900022/
    for fdir in float_dirs:
        fdir_url = urljoin(base_url, fdir)
        # list .nc files in float root
        for fname in list_links(fdir_url, r'\.nc$'):
            full_url = urljoin(fdir_url, fname)
            print(f"🔎 Found file: {full_url}")
            yield full_url
        # go into profiles/
        if "profiles/" in list_links(fdir_url):
            prof_url = urljoin(fdir_url, "profiles/")
            for fname in list_links(prof_url, r'\.nc$'):
                full_url = urljoin(prof_url, fname)
                print(f"🔎 Found profile file: {full_url}")  
                yield full_url

# ----------------------- Chroma + Embeddings -----------------------
client = CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database="flask_rag_db"
)

collection = client.get_or_create_collection("argo_data")
emb_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ----------------------- Extract NC Data -----------------------
def extract_nc_data(file_url):
    response = requests.get(file_url)
    response.raise_for_status()
    docs, metas = [], []

    with Dataset("inmemory.nc", mode="r", memory=response.content) as ds:
        metadata = {}
        doc_text = f"Profile from {file_url}\n"

        for var_name, var_obj in ds.variables.items():
            try:
                arr = var_obj[:]
                if isinstance(arr, np.ma.MaskedArray):
                    arr = arr.filled(np.nan)
                arr = np.array(arr).flatten()

                # Only keep numeric data
                if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                    arr = arr[~np.isnan(arr)]
                    if arr.size == 0:
                        continue

                    stats = {
                        "mean": float(np.mean(arr)),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                        "std": float(np.std(arr)),
                    }

                    # Add to text summary
                    doc_text += (
                        f"{var_name}: {arr.size} pts, "
                        f"range {stats['min']}–{stats['max']}, "
                        f"mean {stats['mean']}, std {stats['std']}\n"
                    )

                    # Save in metadata dict
                    for k, v in stats.items():
                        metadata[f"{var_name}_{k}"] = v

                else:
                    # If categorical or non-numeric, just save sample values
                    if arr.size > 0:
                        metadata[var_name] = str(arr[0])

            except Exception as e:
                print(f"⚠️ Skipping {var_name} in {file_url}: {e}")

        docs.append(doc_text)
        metas.append(metadata)

    return docs, metas


# ----------------------- Main -----------------------
def main():
    nc_urls = list(recursive_nc_files(BASE_URL))
    print(f"🌊 Found {len(nc_urls)} .nc files total.")

    # ⚡ Limit to first 2000 files (change number as needed)
    MAX_FILES = 100
    nc_urls = nc_urls[:MAX_FILES]
    print(f"📉 Limiting run to {len(nc_urls)} files.")

    doc_counter = 0
    for i, url in enumerate(nc_urls, 1):
        print(f"📂 [{i}/{len(nc_urls)}] Processing: {url}", flush=True)
        try:
            docs, metas = extract_nc_data(url)
            if not docs:
                print(f"⚠️ Skipped empty: {url}")
                continue

            ids = [f"doc_{doc_counter+j}" for j in range(len(docs))]
            embeddings = emb_model.embed_documents(docs)
            collection.add(ids=ids, embeddings=embeddings, metadatas=metas, documents=docs)
            doc_counter += len(docs)

            print(f"✅ Stored {len(docs)} doc(s). Total so far: {doc_counter}", flush=True)

        except Exception as e:
            print(f"⚠️ Error {url}: {e}")

    print(f"🎉 Done: {collection.count()} profiles stored in Chroma.")


if __name__ == "__main__":
    main()
