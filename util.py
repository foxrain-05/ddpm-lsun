import lmdb
import hashlib
import requests
import os
from tqdm import tqdm
import zipfile

def download(url, filename, chunk_size= 1024*1024*500):
    with requests.get(url, stream=True, timeout=30) as req:
        total_size = int(req.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, ncols=120, desc=f"{filename:15}")
        
        with open(filename, 'wb') as f:
            for chunk in req.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                progress_bar.update(len(chunk))
            progress_bar.close()



def integrity_check(filename, md5file, chunk_size=1024*1024*500):
    with open(md5file) as f:
        md5, _ = f.read().split()

    md5_hash = hashlib.md5()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data: 
                break
            md5_hash.update(data)

    file_md5 = md5_hash.hexdigest()

    return True if file_md5 == md5 else False


def load_lmdb(filename, out_dir):
    print(f"Exporting {filename} LMDB file to {out_dir} directory.")
    env = lmdb.open(filename, map_size=1099511627776, max_readers=100, readonly=True)

    count = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            image_path = os.path.join(out_dir, key.decode('ascii') + '.jpg')

            with open(image_path, 'wb') as f:
                f.write(value)
            count += 1

            if count % 1000 == 0:
                print(f"Exported {count} files.")

def unzip(filename, out_dir):
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(out_dir)

if __name__ == "__main__":
    pass