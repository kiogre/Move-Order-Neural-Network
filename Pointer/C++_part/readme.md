# What do you need
For the moment I just did the translation from pytohn to C++ of the neural network, for this you need the library onnxruntime-linux-x64-gpu-1.17.1
For the future I already installed the library chess-library, super useful for when I'll need to obtain the next moves.

# Where to execute the files
I created a CMakeList.txt where you basically have to enter to a directory called build (if there isn't create it), where there would be all the different file you compute/execute.
After you are in the directory build, in the terminal you should type:

- cmake ..
- make

And this creates the file you wanted, for the moment I just did such that it works only for test.cpp