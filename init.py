import os
from util import download, integrity_check, unzip, load_lmdb

# Download the dataset
# 데이터 셋 다운로드

# download the dataset from http://dl.yf.io/lsun/objects/cat.zip
# http://dl.yf.io/lsun/objects/cat.zip 에서 데이터 셋을 다운로드 할 수 있습니다.

def download_dataset():
    os.makedirs('data', exist_ok=True)

    if not os.path.exists('data/cat.zip'):
        raise UserWarning('dataset not found. put your dataset in data/ folder. 데이터 셋을 찾을 수 없습니다. data 폴더에 데이터 셋을 넣어주세요.')

    md5_url = 'http://dl.yf.io/lsun/objects/cat.zip.md5'
    if not os.path.exists('data/cat.zip.md5'):
        download(md5_url, 'data/cat.zip.md5')

    if not integrity_check('data/cat.zip', 'data/cat.zip.md5'):
        os.remove('data/cat.zip')
        os.remove('data/cat.zip.md5')
        raise UserWarning("Dataset corrupted. try again. 데이터 셋이 손상 되었습니다. 다시 시도해주세요.")
    else:
        print("Dataset is ready to use. 데이터 셋이 준비 되었습니다.")

    if not os.path.exists('data/cat'):
        unzip('data/cat.zip', 'data/')
    os.remove('data/cat.zip')
    os.remove('data/cat.zip.md5')

    load_lmdb('data/cat', 'data/cat/')
    os.remove('data/cat/data.mdb')
    os.remove('data/cat/lock.mdb')

if __name__ == "__main__":
    download_dataset()
