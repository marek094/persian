#!/bin/bash


( 
    cd scripts;
    make all
)

for f in boundary/*_; 
do 
    python scripts/subset_featspace_to_persistence.py \
        --subset 500 \
        --maxdim 3 \
        --folder $f \
        -j 6 && break
done