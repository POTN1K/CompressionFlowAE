# Flow Compression
____
In recent years the amount of data researchers can generate and access skyrocketed. This introduced
a significant variety in the usefulness of the measured data, posing problems of redundancy and com-
putational expense. This project aims to compress two dimensional Kolmogorov type flow data into a
smaller set of features, while preserving the main properties to a suﬀicient level of accuracy. The results
can contribute to improved flow data storage and flow analysis in a reduced latent space.

This repository is the product of the 3 month Capstone project (TI3150TU), part of the minor Engineering with AI of the TU Delft. The repository contains the FlowCompression module developed by the team, instructional jupyter notebooks, and this readme.

### Contributors, Nov 2022 - Jan 2023:

- Nikolaus Ricker @POTN1K
- Jan Grobusch @JFGrobusch
- Marco Zambolin @mdotzdot
- Ties Harland @tiesharland
- Merlijn Broekers @MerlijnBroekers2
- Aron Domotor Kohlheb @Dome-Kohlheb



## Module Architecture
___

The implementation of models relies heavily on the Model parent class, 
which defines common logic methods and most user facing functionality (such as <code>model.performance()</code>,
<code>model.paththrough()</code>, etc.).

The POD and AE subclasses inherit these common methods.
The AE subclass is extended to facilitate the physical 
loss function and heirarchical AE. 

### Saving (& Loading) Directories
Trained models are saved in the KerasModels folder.

    FlowCompression
    ├ ...
    ├ KerasModels
        ├ HeirarchicalAE
            {
            autoencoder_h.h5
            encoder_h.h5
            decoder_h.h5
            }
        ├ StandardDims
            ├ StandardAE_dim=1
            ├ ...
        ├ Raw
            ├ FreshModel_01
            ├ ...
    ├ ...

Loading is described in the AE Jupyter file. 
A single model requires saving all 3 components
(autoencoder, encoder, decoder). Provided is trained
Heirarchical AE, Phyiscal (loss function) AE, and a set of
standard AE's of latent space dimension 1 to 64. 
Further models trained using 
<code>model.train_test_batch(save=True)</code>
are by default saved to the Raw directory. 
Manual saving defaults to your working directory.

### Performance Directory
Performance evaluated by <code>model.train_test_batch()</code>
is archived in the <code>TuningDivision</code> folder. As with 
model saving, there is a default <code>Raw</code> subdirectory to
avoid clutter.

### Required Libraries
The environment can be installed using the following lines of code 
using conda or pip. It should be run using python 3.10 or higher.

<code>
absl-py==1.3.0
asttokens==2.2.1
astunparse==1.6.3
backcall==0.2.0
cachetools==5.2.0
certifi==2022.12.7
charset-normalizer==2.1.1
colorama==0.4.6
comm==0.1.2
contourpy==1.0.6
cycler==0.11.0
debugpy==1.6.6
decorator==5.1.1
entrypoints==0.4
executing==1.2.0
flatbuffers==22.12.6
fonttools==4.38.0
gast==0.4.0
google-auth==2.15.0
google-auth-oauthlib==0.4.6
google-pasta==0.2.0
grpcio==1.51.1
h5py==3.7.0
idna==3.4
ipykernel==6.20.2
ipython==8.8.0
jedi==0.18.2
joblib==1.2.0
jupyter_client==7.4.9
jupyter_core==5.1.5
keras==2.11.0
kiwisolver==1.4.4
libclang==14.0.6
Markdown==3.4.1
MarkupSafe==2.1.1
matplotlib==3.6.2
matplotlib-inline==0.1.6
nest-asyncio==1.5.6
numpy==1.23.5
oauthlib==3.2.2
opt-einsum==3.3.0
packaging==23.0
parso==0.8.3
pickleshare==0.7.5
Pillow==9.3.0
platformdirs==2.6.2
prompt-toolkit==3.0.36
protobuf==3.19.6
psutil==5.9.4
pure-eval==0.2.2
pyasn1==0.4.8
pyasn1-modules==0.2.8
Pygments==2.14.0
pyparsing==3.0.9
python-dateutil==2.8.2
pywin32==305
pyzmq==25.0.0
requests==2.28.1
requests-oauthlib==1.3.1
rsa==4.9
scikit-learn==1.1.3
scipy==1.9.3
six==1.16.0
stack-data==0.6.2
tensorboard==2.11.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow==2.11.0
tensorflow-estimator==2.11.0
tensorflow-intel==2.11.0
tensorflow-io-gcs-filesystem==0.28.0
termcolor==2.1.1
threadpoolctl==3.1.0
tornado==6.2
traitlets==5.8.1
typing_extensions==4.4.0
urllib3==1.26.13
wcwidth==0.2.6
Werkzeug==2.2.2
wrapt==1.14.1
</code>