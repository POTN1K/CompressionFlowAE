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