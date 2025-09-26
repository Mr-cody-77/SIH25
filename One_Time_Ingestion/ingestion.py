import os
import re
import requests
import numpy as np
from urllib.parse import urljoin
from netCDF4 import Dataset
from chromadb import CloudClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
import random

load_dotenv()

BASE_URL = "https://data-argo.ifremer.fr/dac/aoml/"

def list_links(url, pattern=None, retries=3, backoff=2):
    """Return all href links from a directory URL, optionally filtered by regex."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=20)  # ⏳ timeout in seconds
            resp.raise_for_status()
            links = re.findall(r'href="([^"]+)"', resp.text)
            if pattern:
                links = [l for l in links if re.search(pattern, l)]
            return links
        except requests.exceptions.RequestException as e:
            wait = backoff * (2 ** attempt) + random.random()
            print(f"⚠️ Error fetching {url} (attempt {attempt+1}/{retries}): {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    print(f"❌ Failed to fetch {url} after {retries} retries.")
    return []

def recursive_nc_files(base_url, limit=500):
    """Crawl float directories and yield full URLs of .nc files (limited)."""
    count = 0
    float_dirs = list_links(base_url, r'^[0-9]+/$')   # e.g. 1900022/

    for fdir in float_dirs:
        fdir_url = urljoin(base_url, fdir)

        # list .nc files in float root
        for fname in list_links(fdir_url, r'\.nc$'):
            full_url = urljoin(fdir_url, fname)
            print(f"🔎 Found file: {full_url}")
            yield full_url
            count += 1
            if count >= limit:
                return

        # go into profiles/
        if "profiles/" in list_links(fdir_url):
            prof_url = urljoin(fdir_url, "profiles/")
            for fname in list_links(prof_url, r'\.nc$'):
                full_url = urljoin(prof_url, fname)
                print(f"🔎 Found profile file: {full_url}")
                yield full_url
                count += 1
                if count >= limit:
                    return

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

        # ✅ Try to extract lat/lon/time if present
        for key in ["LATITUDE", "LONGITUDE", "JULD"]:
            if key in ds.variables:
                try:
                    val = ds.variables[key][:]
                    if isinstance(val, np.ma.MaskedArray):
                        val = val.filled(np.nan)
                    val = float(np.array(val).flatten()[0])
                    metadata[key.lower()] = val
                except Exception:
                    pass

        # ✅ Only keep summaries for 3–5 key variables to avoid quota issues
        keep_vars = ["TEMP", "PSAL", "PRES"]  # Temperature, Salinity, Pressure
        for var_name in ds.variables:
            if var_name not in keep_vars:
                continue
            try:
                arr = ds.variables[var_name][:]
                if isinstance(arr, np.ma.MaskedArray):
                    arr = arr.filled(np.nan)
                arr = np.array(arr).flatten()

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

                    doc_text += (
                        f"{var_name}: {arr.size} pts, "
                        f"range {stats['min']}–{stats['max']}, "
                        f"mean {stats['mean']}, std {stats['std']}\n"
                    )

                    # ✅ Only store mean to reduce metadata keys
                    metadata[f"{var_name.lower()}_mean"] = stats["mean"]

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

"""
def main(limit=None):
    # Crawl .nc files
    nc_urls = list(recursive_nc_files(BASE_URL, limit=limit))
    print(f"🌊 Found {len(nc_urls)} .nc files total (may include old ones).")

    existing = set(collection.get(ids=None)["ids"])  # fetch existing IDs
    print(f"📦 Already have {len(existing)} docs in Chroma.")

    new_urls = [u for u in nc_urls if u not in existing]
    print(f"✨ {len(new_urls)} new files to add.")

    doc_counter = 0
    for url in new_urls:
        print(f"📂 Processing: {url}")
        try:
            docs, metas = extract_nc_data(url)
            if not docs:
                continue

            ids = [url]  # ✅ use file URL as unique ID
            embeddings = emb_model.embed_documents(docs)
            collection.add(ids=ids, embeddings=embeddings, metadatas=metas, documents=docs)
            doc_counter += len(docs)
        except Exception as e:
            print(f"⚠️ Error {url}: {e}")

    print(f"🎉 Update complete: {collection.count()} total profiles stored in Chroma.")

"""