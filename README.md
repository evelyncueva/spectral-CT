### Data & codes: Synergistic multi-spectral CT reconstruction with dTV
**Authors:** Evelyn Cueva & Matthias J. Ehrhardt


#### 1. Read and/or generate data

For real  data (bird dataset), use the file `real_generate_data.py` to read data, reconstruct reference images and to reconstruct side information. The values of regularizer parameters $\alpha$, the number of angles, detectors and reconstruction size are specified within the file. The resulting `npy` files are saved in the folder `data` in the subfolder related to dataset name `bird`.

For synthetic data (geometric dataset), use the file `synthetic_generate_data.py` to create the synthetic data, save reference images and reconstruct side information for different values of regularization parameter $\alpha$. As before, the resulting data files are saved in `data/geometric` path. 

Before run any experiment (explained in section below), we have to decide the data and reconstruction sizes, since references and side information need to be generated under those specifications, *i.e.*, if we want to reconstruct a $512\times 512$ image using a sinogram of size $90\times 552$, we have to generate sinograms, references and side information using these dimensions. 

#### 2. Run experiments
To run one experiment, we first need to decided the following parameters:

1. name of dataset: geometric or bird
2. reconstruction size: an integer, *e.g.* 512
3. number of view angles in the data space, *e.g.* 90
4. number of detectors in the data space, *e.g.* 552
5. $\eta$  parameter in dTV definition, *e.g.* 1e-2
6. regularization parameter $\alpha$ to be used in the reconstruction 
7. reference type, we always choose a *TV* type
8. parameter $\alpha$ used to reconstruct side information in string format, *e.g.* 0_0001
9. name of algorithm to be run: 
   - *no_prior_1v:* no regularizer with non-negativity constrain
   - *fbsTV:* forward backward splitting with TV
   - *fbsdTV:* forward backward splitting with dTV
   - *bregmanTV:* bregman iterations with TV
   - *bregmandTV:* bregman iterations with dTV
   - *all_fbs:* for both, fbsTV and fbsdTV
   - *all_bregman:* for both, bregmanTV and bregmandTV
   - *all:* to run the five algorithms, no_prior_1, fbsTV, fbsdTV, bregmanTV, bregmandTV
10. energy channel to be reconstructed
   - E0
   - E1
   - E2
   - all: for E0, E1 and E2   

Then, we edit the file `msct_run_local.sh` to set the values of the parameters that we need to run, for example, the lines below run the `msct_main.py` file for synthetic data (geometric) to reconstruct an image of size $512\times 512$ using a sinogram of size $90\times 552$ with $\eta=0.01$ and $\alpha=0.05$. Here, we specify that the side information to be used comes from the solution of forward-backward splitting algorithm with TV regularization using $\alpha=1\times10^{-4}$. We will use Bregman iterations to reconstruct the $E_2$ channel.

`python msct_main.py geometric 512 90 552 1e-2 5e-2 TV 0_0001 bregmandTV E2`

For this example, the results are saved in the folder `results_npy/geometric_data/d90x552_gt512_u512/npy`, the file name specifies all the parameters that we have chosen previously. Following the same example, the file name is  `E2_bregmandTV_alpha_5.0e_02_tol_1e_05_sinfo_0_0001_eta_1.0e_02_output.npy`. The tolerance is always fixed in $1\times10^{-5}$ and we do not need to specify in the `msct_run_local.sh` file.

#### 3. Plot (paper) figures

Once we have all the files to show the results, we run the `jupyter notebook` files: `article_figures_real.ipynb` and `article_figures_synthetic.ipynb` for real and synthetic data, respectively. These notebooks generate all the graphics included in the paper. If the images are related with data, they are saved in folders `bird_images` or `geometric_images`, depending on dataset name. And if the images are related with the reconstruction process, they are saved in the same path in the folder `figures`. 
