
# Can Deep Learning Relax Endomicroscopy Hardware Miniaturization Requirements?

# Abstract

Confocal laser endomicroscopy (CLE) is a novel imaging modality that provides in vivo histological cross-sections of examined tis- sue. Recently, attempts have been made to develop miniaturized in vivo imaging devices, specifically confocal laser microscopes, for both clinical and research applications. However, current implementations of minia- ture CLE components, such as confocal lenses, compromise image resolu- tion, signal-to-noise ratio, or both, which negatively impacts the utility of in vivo imaging. In this work, we demonstrate that software-based tech- niques can be used to recover lost information due to endomicroscopy hardware miniaturization and reconstruct images of higher resolution. Particularly, a densely connected convolutional neural network is used to reconstruct a high-resolution CLE image from a low-resolution input. In the proposed network, each layer is directly connected to all subsequent layers, which results in an effective combination of low-level and high- level features and efficient information flow throughout the network. To train and evaluate our network, we use a dataset of 181 high-resolution CLE images. Both quantitative and qualitative results indicate superior- ity of the proposed network compared to traditional interpolation tech- niques and competing learning-based methods. This work demonstrates that software-based super-resolution is a viable approach to compensate for loss of resolution due to endoscopic hardware miniaturization.

# Keywords
Confocal Laser Endomicroscopy, Super-resolution, Deep learning

# Cite
If you use our code, please cite our paper: 
[Can Deep Learning Relax Endomicroscopy Hardware Miniaturization Requirements?](https://www.cs.sfu.ca/~hamarneh/ecopy/miccai2018c.pdf)

The corresponding bibtex entry is:

```
@inproceedings{izadi2018can,
  title={Can deep learning relax endomicroscopy hardware miniaturization requirements?},
  author={Izadi, Saeed and Moriarty, Kathleen P and Hamarneh, Ghassan},
  booktitle={International conference on medical image computing and computer-assisted intervention},
  pages={57--64},
  year={2018},
  organization={Springer}
}
```
