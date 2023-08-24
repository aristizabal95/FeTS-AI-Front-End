FROM ghcr.io/fets-ai/fetstool_docker_dependencies:0.0.2.gpu

LABEL authors="FeTS_Admin <admin@fets.ai>"

RUN apt-get update && apt-get update --fix-missing && apt-get install -y libnss3 libnspr4 libxcursor1 libxcursor-dev libasound2 libdbus-1-dev libglfw3-dev libgles2-mesa-dev ffmpeg libsm6 libxext6 python3.8 python3.8-venv python3.8-dev python3-setuptools

# older python
RUN apt-get update -y && apt install -y --reinstall software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt update -y && apt install -y python3.7 python3.7-venv python3.7-dev python3-setuptools

ENV PATH=/workspace/CaPTk/bin/qt/5.12.1/bin:/workspace/CaPTk/bin/qt/5.12.1/libexec:$PATH
ENV CMAKE_PREFIX_PATH=/workspace/CaPTk/bin/ITK-build:/workspace/CaPTk/bin/DCMTK-build:/workspace/CaPTk/bin/qt/5.12.1/lib/cmake/Qt5:$CMAKE_PREFIX_PATH

RUN pwd && ls -l

WORKDIR /Front-End

COPY . .

RUN pwd && ls -l && mv ./data/Algorithms_for_fetsTool1.0.zip OpenFederatedLearning/submodules/fets_ai/ && cd OpenFederatedLearning/submodules/fets_ai/ && unzip -qq Algorithms_for_fetsTool1.0.zip && rm -rf Algorithms_for_fetsTool1.0.zip

RUN pwd && ls -l && mv ./data/GANDLF_for_fetsTool1.0.zip OpenFederatedLearning/submodules/fets_ai/Algorithms && cd OpenFederatedLearning/submodules/fets_ai/Algorithms && unzip -qq GANDLF_for_fetsTool1.0.zip && rm -rf GANDLF_for_fetsTool1.0.zip

## C++ build
RUN mkdir bin && cd bin && cmake -DCMAKE_INSTALL_PREFIX="./install/appdir/usr" -DITK_DIR="/workspace/CaPTk/bin/ITK-build" -DDCMTK_DIR="/workspace/CaPTk/bin/DCMTK-build" -DBUILD_TESTING=OFF .. && make -j$(nproc) && make install/strip 

# ## Python package installation -- this is for the new docker image, which is much simpler
# RUN cd bin/install/appdir/usr/bin/ && python3.8 -m venv ./venv && ./venv/bin/pip install --upgrade pip wheel && ./venv/bin/pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu && ./venv/bin/pip install -e . && ./venv/bin/pip install setuptools-rust Cython scikit-build scikit-learn openvino-dev==2023.0.1 && ./venv/bin/pip install -e .

# set up environment and install correct version of pytorch
RUN echo "Setting up virtual environment for OpenFederatedLearning with base dependencies" && \
    cd bin/install/appdir/usr/bin/OpenFederatedLearning && \
    rm -rf ./venv && python3.7 -m venv ./venv && ./venv/bin/pip install Cython && \
    ./venv/bin/pip install --upgrade pip setuptools wheel setuptools-rust && \
    ./venv/bin/pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html && \
    ./venv/bin/pip install wheel && \
    ./venv/bin/pip install SimpleITK==1.2.4 && \
    ./venv/bin/pip install protobuf==3.17.3 grpcio==1.30.0 && \
    ./venv/bin/pip install opencv-python==4.2.0.34 && \
    ./venv/bin/pip install scikit-build scikit-learn && \
    make install_openfl && \
    make install_openfl_pytorch

RUN echo "Setting up virtual environment for OpenFederatedLearning with second-level dependencies" && \
    cd bin/install/appdir/usr/bin/OpenFederatedLearning && \
    ./venv/bin/pip install ../BrainMaGe && \
    ./venv/bin/pip install ./submodules/fets_ai/Algorithms && \
    ./venv/bin/pip install -e ./submodules/fets_ai/Algorithms/GANDLF

RUN echo "Installing separate environment for LabelFusion" && \
    cd bin/install/appdir/usr/bin/LabelFusion && \
    rm -rf venv && python3.8 -m venv ./venv && \
    ./venv/bin/pip install --upgrade pip setuptools wheel setuptools-rust && \
    ./venv/bin/pip install -e .

RUN echo "Downloading model weights" && \
    cd bin/install/appdir/usr/data/fets && \
    wget https://upenn.box.com/shared/static/f7zt19d08c545qt3tcaeg7b37z6qafum.zip -O nnunet.zip && \
    unzip -qq nnunet.zip && rm -rf nnunet.zip && \
    wget https://upenn.box.com/shared/static/hhvn8nb9xtz6nxcilmdl8kbx9n1afkdu.zip -O ./fets_consensus_models.zip && \
    unzip -qq fets_consensus_models.zip && rm -rf fets_consensus_models.zip
    
### put together a data example that is already aligned and ready to invoke the brain extraction and tumor segmentation

# set up the docker for GUI
ENV LD_LIBRARY_PATH=/CaPTk/bin/qt/5.12.1/lib:$LD_LIBRARY_PATH
ENV PATH=/Front-End/bin/install/appdir/usr/bin/:$PATH
ENV QT_X11_NO_MITSHM=1
ENV QT_GRAPHICSSYSTEM="native"

RUN echo "Env paths\n" && echo $PATH && echo $LD_LIBRARY_PATH

# define entry point
ENTRYPOINT ["/Front-End/bin/install/appdir/usr/bin/FeTS_CLI_Segment"]
