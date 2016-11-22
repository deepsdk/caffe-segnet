#!/bin/bash


scp ./build/tools/classification.bin lambda:~/package/bin/classification_segnet.bin

scp ./build/lib/libcaffe_segnet.so lambda:~/package/lib
