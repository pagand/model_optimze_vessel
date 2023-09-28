#!/usr/bin/env bash

cd data
export LICID=15SC-g_xBDa8WpBVXU8RDTPtQ4yXzjA7E
export FILEID=1ISveXA1FDV_jToLHDY9tp-Ghsk6UYcaK

# Download license file
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${LICID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${LICID}" -O LICENSE.txt && rm -rf /tmp/cookies.txt



# Download RL dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${FILEID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O rl_data.pkl && rm -rf /tmp/cookies.txt


