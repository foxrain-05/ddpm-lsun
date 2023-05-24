import os
from util import download, integrity_check
import zipfile

# Download the dataset

data_url = 'http://dl.yf.io/lsun/objects/cat.zip'
md5_url = 'http://dl.yf.io/lsun/objects/cat.zip.md5'

os.makedirs('data', exist_ok=True)

if not os.path.exists('data/cat.zip'):
    download(data_url, 'data/cat.zip')

if not os.path.exists('data/cat.zip.md5'):
    download(md5_url, 'data/cat.zip.md5')

if not integrity_check('data/cat.zip', 'data/cat.zip.md5'):
    os.remove('data/cat.zip')
    os.remove('data/cat.zip.md5')
    raise UserWarning("Dataset corrupted. try again. 데이터 셋이 손상 되었습니다. 다시 시도해주세요.")
else:
    print("Dataset is ready to use. 데이터 셋이 준비 되었습니다.")


