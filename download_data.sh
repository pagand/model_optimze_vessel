#!/usr/bin/env bash

cd data

# Download license file
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=15SC-g_xBDa8WpBVXU8RDTPtQ4yXzjA7E' -O LICENSE.txt

# Download RL dataset
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1UnlG5t0fJHceXTY7j8GiUWD6reoxsY6S' -O rl_data.pkl

