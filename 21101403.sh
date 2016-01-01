#!/bin/bash
nvcc -o dot_product dot_product.cu
for k in 1000000 5000000 10000000; do for b in 32 64 128 256 512; do ./dot_product $k $b; done; done;
