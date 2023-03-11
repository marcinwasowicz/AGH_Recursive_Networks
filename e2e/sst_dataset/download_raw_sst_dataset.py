import urllib.request
import sys
import os
import zipfile
import shutil


def download(url, dirpath):
    filename = url.split("/")[-1]
    filepath = os.path.join(dirpath, filename)
    try:
        u = urllib.request.urlopen(url)
    except:
        print("URL %s failed to open" % url)
        raise Exception
    try:
        f = open(filepath, "wb+")
    except:
        print("Cannot write %s" % filepath)
        raise Exception
    try:
        filesize = int(u.headers["Content-Length"])
    except:
        print("URL %s failed to report length" % url)
        raise Exception
    print("Downloading: %s Bytes: %s" % (filename, filesize))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print("")
            break
        else:
            print("", end="\r")
        downloaded += len(buf)
        f.write(buf)
        status = ("[%-" + str(status_width + 1) + "s] %3.2f%%") % (
            "=" * int(float(downloaded) / filesize * status_width) + ">",
            downloaded * 100.0 / filesize,
        )
        print(status, end="")
        sys.stdout.flush()
    f.close()
    return filepath


def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)


def download_sst(dirpath):
    if os.path.exists(dirpath):
        print("Found SST dataset - skip")
        return
    url = "http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip"
    parent_dir = os.path.dirname(dirpath)
    unzip(download(url, parent_dir))
    os.rename(
        os.path.join(parent_dir, "stanfordSentimentTreebank"),
        os.path.join(parent_dir, "sst"),
    )
    shutil.rmtree(os.path.join(parent_dir, "__MACOSX"))  # remove extraneous dir


if __name__ == "__main__":
    base_dir = "./"

    # data
    data_dir = os.path.join(base_dir, "data")
    sst_dir = os.path.join(data_dir, "sst")

    # download dependencies
    download_sst(sst_dir)
