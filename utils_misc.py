import numpy as np, os
import requests

# GPU-related business
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_smi')
    memory_available = [int(x.split()[2])+5*i for i, x in enumerate(open('tmp_smi', 'r').readlines())]
    os.remove("tmp_smi")
    return np.argmax(memory_available)

def select_freer_gpu():
    freer_gpu = str(get_freer_gpu())
    print("Will use GPU: %s" % (freer_gpu))
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""+freer_gpu
    return freer_gpu

# Google Drive related
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
