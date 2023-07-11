import requests
from tqdm import tqdm
import tarfile

def download_list(url_list):
    for url in url_list:
        path = url.split('/')[-1]
        r = requests.get(url, stream=True)
        with open(path, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))

            for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
                if chunk:
                    f.write(chunk)
    print('All dataset files downloaded!')

download_list(['http://codh.rois.ac.jp/kmnist/dataset/kkanji/kkanji.tar'])

with tarfile.open("kkanji.tar") as tar:
    tar.extractall("test")