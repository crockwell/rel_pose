# The 8-Point Algorithm as an Inductive Bias for Relative Pose Prediction by ViTs (3DV 2022)

<img src="docs/teaser.png" align="right" alt="drawing" width="50%">

Chris Rockwell, Justin Johnson and David F. Fouhey

**[[Project Website](https://crockwell.github.io/rel_pose/)] [[Paper](https://crockwell.github.io/rel_pose/data/paper.pdf)] 
[[Supplemental](https://crockwell.github.io/rel_pose/data/supp.pdf)]**

We propose three small modifications to a ViT via the Essential Matrix Module, enabling computations similar to the
Eight-Point algorithm. The resulting mix of visual and positional features is a good inductive bias for pose estimation.

### Installation and Demo
- [Install](docs/INSTALL.md)
- [Demo](docs/DEMO.md)

### Training and Evaluation
- [RealEstate10K](docs/REALESTATE.md)
- [Matterport](docs/MATTERPORT.md)

### Citation
If you use this code for your research, please consider citing:
```
@inProceedings{Rockwell2021,
  author = {Chris Rockwell and Justin Johnson and David F. Fouhey},
  title = {The 8-Point Algorithm as an Inductive Bias for Relative Pose Prediction by ViTs},
  booktitle = {3DV},
  year = 2022
}
```

### Special Thanks
Thanks to <a href="https://jinlinyi.github.io/">Linyi Jin</a>, <a href="https://www.cs.cornell.edu/~ruojin/">Ruojin Cai</a> and <a href="https://zachteed.github.io/">Zach Teed</a> for help replicating and building upon their works. Thanks to <a href="https://mbanani.github.io/">Mohamed El Banani</a>, <a href="http://kdexd.xyz/">Karan Desai</a> and <a href="https://nileshkulkarni.github.io/">Nilesh Kulkarni</a> for their many helpful suggestions. Thanks to Laura Fink and UM DCO for their tireless support with computing!
