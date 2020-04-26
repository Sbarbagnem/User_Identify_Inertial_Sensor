#!/usr/bin/env bashù
for dataset in "unimib" "sbahr" "realdisp"
do
    for i in {0..1}
    do
    python main.py -d $dataset -f $i -m 2
    done
done