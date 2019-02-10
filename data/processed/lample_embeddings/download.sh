#!/bin/sh

echo
echo
echo -e "If you have not installed gensim, please run \"pip install gensim\""
echo
echo

filename="skip100.txt"
file_id="0B23ji47zTOQNUXYzbUVhdDN2ZmM"
query=`curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=${file_id}" \
| perl -nE'say/uc-download-link.*? href="(.*?)\">/' \
| sed -e 's/amp;//g' | sed -n 2p`
url="https://drive.google.com$query"
echo $url
curl -b ./cookie.txt -L -o ${filename} $url
