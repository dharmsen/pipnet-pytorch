PIP-net (CVPR 2023) PyTorch extension 
---

Making PIPNet more extensible and flexible.


## How to start

1. Use the ```make init``` command, this will install the venv and all required dependencies.
2. Source the virtual environment with ```source .venv/bin/activate```
3. Download the [CUB_200_2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) dataset into a ```data``` directory. 
4. Download pretrained weights from the main repository, [here](https://github.com/M-Nauta/PIPNet?tab=readme-ov-file#training-pip-net).
5. Use the ```make train``` command to start the training process.

Please note that this is still WIP, there will be changes in how the training can be performed.
This is the version that can get you up and running the quickest.


## Contributing

Dev setup steps and general pointers.
1. Run the ```make init_dev``` command, this will install and setup necessary pre-commit hooks.
2. Read the paper. Besides the fact that it is a cool paper, if you want to contribute get to know the system you will help improve.
3. There are `make format` and `make check` commands that will perform formatting and code inspection, the tool used is `ruff`. 
During development use these often, same checks run in pre-commit to insure consistent standards in the repository.
4. Each change must be completely tested. Unit tests are mandatory, integration tests must be written if change impacts different modules of this package.

## Credits

Main paper and codebase:
```
@inproceedings{nauta2023pip,
  title={Pip-net: Patch-based intuitive prototypes for interpretable image classification},
  author={Nauta, Meike and Schl{\"o}tterer, J{\"o}rg and Van Keulen, Maurice and Seifert, Christin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2744--2753},
  year={2023}
}
```

CUB dataset:
```
@techreport{WahCUB_200_2011,
	Title = ,
	Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
	Year = {2011}
	Institution = {California Institute of Technology},
	Number = {CNS-TR-2011-001}
}
```
