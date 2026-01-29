# Segmentation branch

This branch is customized for segmentation of 4DGS.

# Installation

```bash
# skip this if you have installed the environment for original 4DGS
conda env create --file environment.yml
conda activate 4dgs

# additional packages
pip install hdbscan imageio
pip install pytorch3d==0.7.1
pip install ./diff-gaussian-rasterization_contrastive_f
pip install third_party/segment-anything
```

## Prepare Data

Suppose we have trained a Gaussian model under `./output/N3V/sear_steak`, then
following [SAGA](https://github.com/Jumpat/SegAnyGAussians) to prepare the data and SAM checkpoint required for segmentation.

For example,
```bash
python extract_segment_everything_masks.py --image_root <path to the scene data> --sam_checkpoint_path <path to the pre-trained SAM model> --downsample <1/2/4/8>
python get_scale.py --image_root <path to the scene data> --model_path ./output/N3V/sear_steak
```

## Running

Train 4D Gaussian Affinity Features
```bash
python train_contrastive_feature.py -m ./output/N3V/sear_steak --iterations 10000 --num_sampled_rays 1000
```

Visualize the segmentation
```bash
python vis_segment.py
```

## Acknowledgement
The implementation refers to [SAGA](https://github.com/Jumpat/SegAnyGAussians), and we sincerely thank them for their contributions to the community.
