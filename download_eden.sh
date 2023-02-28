#!/bin/sh -f
#download EDEN-sample dataset: https://lhoangan.github.io/eden/
dir=data/EDEN

# data folder
mkdir -p $dir

# download depth data: 7.7GB
wget https://isis-data.science.uva.nl/hale/EDEN-samples/Depth.zip -P $dir
mkdir -p ${dir}/Depth
unzip ${dir}/Depth.zip -d ${dir}/Depth

# download RGB data: 11GB
wget https://isis-data.science.uva.nl/hale/EDEN-samples/RGB.zip -P $dir
mkdir -p ${dir}/RGB
unzip ${dir}/RGB.zip -d ${dir}/RGB
