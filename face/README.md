### this folder holds toy like examples of detecting face and recognizing the face from a web cam feed

Pre-requisite is to build dlib to leverage the NVidia CUDA capable card your your computer. If you do not use CUDA, everything will be so slow!

You will also need to install `pip install opencv-contrib-python`

First step, is to run `python enc.py no_space_name` and hit `s` when you are ready to generate encoding of your face. Hit `RETURN` when done and this will save an entry under `./encoding`

I am trying out various tweaks of the parameters, but doing less than 5 captures is good as capturing over 50 and averaging the encodings found.

Next, run `python face.py` and play around with hitting s , b , f and RETURN/esc to exit
