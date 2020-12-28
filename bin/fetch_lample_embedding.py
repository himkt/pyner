import pathlib
import re

import requests


def download_file_from_google_drive():
    URL = "https://drive.google.com{}"

    file_id = "0B23ji47zTOQNUXYzbUVhdDN2ZmM"
    session = requests.Session()

    target_url = URL.format("/uc?export=download")
    response = session.get(target_url, params={"id": file_id})
    content = response.content.decode("utf-8")  # NOQA
    pattern = re.compile('<a id="uc-download-link".*? href="(.*?)".*?</a>')

    result = re.search(pattern, content)
    params = result.group(1)
    params = params.replace("amp;", "")

    target_url = URL.format(params)
    response = session.get(target_url, cookies=response.cookies, stream=True)
    save_response_content(response)


def save_response_content(response):
    cur_length = 0
    chunk_size = 32768

    skipngram_path = pathlib.Path("data/external/LampleEmbeddings")
    with open(skipngram_path / "skipngram_100d.txt", "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                cur_length += len(chunk)
                cur_mbytes = cur_length / 1024 ** 2
                print(f"downloaded: {cur_mbytes:.4f} MB\r", end="")
                f.write(chunk)


if __name__ == "__main__":
    download_file_from_google_drive()
