#!/usr/bin/env bash

for dataset in "unimib" "sbhar" "realdisp"
do
    for i in {0..1}
    do
    python main.py -d $dataset -f $i -m 1
    done
done
