#!/bin/bash


( 
    cd scripts;
    make all
)

for f in boundary/*_; 
do 
    python scripts/featspace_to_persistence.py --folder $f -j 6 && break
done