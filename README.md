# coulomb-matrix-and-symmetry-function-on-QM9

This is a small tutorial where I have computed the coulomb matrix and atom-centered symmetry function (ACSF) descriptor on QM9. The ACSF descriptor here is Behler Parrinello SF computed using ```torchani``` https://github.com/aiqm/torchani.

The BPFS descriptor is slightly modified. Instead of using different neural networks for each atom-type we incorporate atom charge in the descriptor. 

```data``` folder contains 3 ```.npz``` files i.e. ```train.npz```, ```valid.npz``` and ```test.npz```. This data was created using https://github.com/vgsatorras/egnn.

The results shown below are based on the following.
Training set = 17748.
Validated set = 13083.

Some of the code is also adopted from my other repo https://github.com/rohitmodee/DART.

![image](https://github.com/rohitmodee/coulomb-matrix-and-symmetry-function-on-QM9/assets/24433906/c4fbc9d9-5464-408b-9ead-2c61b772c9c6)


