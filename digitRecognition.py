import gzip
import numpy as np

with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    content = f.read()

print(type(content))