# learn_tvm

Usage:   
    1. Construct a virtual python env 
    2. Install a nightly version tvm python3 -m  pip install mlc-ai-nightly -f https://mlc.ai/wheels

Common compilation and installtion of standard tvm:
    1. git clone tvm
    2. cp cmake/config.cmake build
    3. cmake .. && make
    4. set env:
        export TVM_HOME=/path/to/tvm
        export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH}
    5. pip3 install IPython && synr
