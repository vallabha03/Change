#paths
from pathlib import Path

root = Path('data/')
img_dir = root / 'images'
train_labels_dir = root / 'train_labels'
test_labels_dir = root / 'test_labels'

#raster_source
from rastervision.core.data import RasterioSource

img_path = str(
    img_dir / 'beirut' /
    'imgs_1/S2A_OPER_MSI_L1C_TL_EPA__20160929T004310_A000833_T36SYC_B04.tif')

# You can actually pass a list of URIs to the constructor if all the images are
# part of a single mosaic. This is not the case here, so we pass only
# a single URI.
raster_source = RasterioSource(uris=[img_path])

#chip_read
from rastervision.core.box import Box

# specify the window to be read in pixel coordinates
window = Box(ymin=0, xmin=0, ymax=400, xmax=400)

# "activate" the RasterioSource, so that it can download the file if needed
# and open it for reading
chip = raster_source.get_chip(window=window)

#display_2dnumpy
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
sns.reset_defaults()

def draw_chip_with_colorbar(chip, ax, clevels=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=0.02)
    if clevels:
        cmap = plt.get_cmap('viridis', clevels)
    else:
        cmap = plt.get_cmap('viridis')
    im = ax.imshow(chip, cmap=cmap)
    fig.colorbar(im, cax=cax, orientation='horizontal')
    ax.axis('off')

#display_single_channel
fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(6, 6))
draw_chip_with_colorbar(chip, ax)
plt.show()
fig.savefig('raster_source_chip.png', bbox_inches='tight', pad_inches=0.)

#chips_from_all_bands
img_paths = sorted((img_dir / 'beirut' / 'imgs_1/').glob('*.tif'))
raster_sources = [RasterioSource(uris=[str(path)]) for path in img_paths]

chips = []
for rs in raster_sources:
    chip = rs.get_chip(window=window)
    chips.append(chip)

band_names = [p.stem[-3:] for p in img_paths]


#Display_all_channels
fig, axs = plt.subplots(3, 5, squeeze=False, figsize=(12, 9))
for ax, chip, name in zip(axs.flat, chips, band_names):
    draw_chip_with_colorbar(chip, ax)
    ax.set_title(f'Band {name}')
    
axs.flat[-1].axis('off')
axs.flat[-2].axis('off')
fig.tight_layout(h_pad=0.5, w_pad=0.5)
plt.show()
fig.savefig('raster_source_chips_unaligned.png', bbox_inches='tight', pad_inches=0.)

#MultirasterSource
from rastervision.core.data import MultiRasterSource

multi_rs = MultiRasterSource(
    # the 13 raster sources that we initialized above
    raster_sources=raster_sources,
    # the combined raster source will act as if it had the extent 
    # of the raster source at this index i.e. band B02
    primary_source_idx=1)

# shape = (400, 400, 13)
chip_multi = multi_rs.get_chip(window=window)

#modified_allbands_display
fig, axs = plt.subplots(3, 5, squeeze=False, figsize=(12, 9))
for ax, chip, name in zip(axs.flat, chip_multi.transpose(2, 0, 1), band_names):
    draw_chip_with_colorbar(chip, ax)
    ax.set_title(f'Band {name}')
    
axs.flat[-1].axis('off')
axs.flat[-2].axis('off')
fig.tight_layout(h_pad=0.5, w_pad=0.5)
plt.show()
fig.savefig('multi_raster_source_chips.png', bbox_inches='tight', pad_inches=0.)

#StastsTransformer
from rastervision.core.raster_stats import RasterStats
from rastervision.core.data.raster_transformer import StatsTransformer

stats = RasterStats()
stats.compute(raster_sources=[multi_rs])

means=stats.means
stds=stats.means
stats_transformer = StatsTransformer(means=means,stds=stds)

multi_rs_normalized = MultiRasterSource(
    raster_sources=raster_sources, 
    primary_source_idx=1, 
    raster_transformers=[stats_transformer])

#pixel_range_(0-255)
# shape = (400, 400, 13)
chip_multi_normalized = multi_rs_normalized.get_chip(window=window)

#Multi_band_255pixels
fig, axs = plt.subplots(3, 5, squeeze=False, figsize=(12, 9))
for ax, chip, name in zip(axs.flat, chip_multi_normalized.transpose(2, 0, 1), band_names):
    draw_chip_with_colorbar(chip, ax)
    ax.set_title(f'Band {name}')
    
axs.flat[-1].axis('off')
axs.flat[-2].axis('off')
fig.tight_layout(h_pad=0.5, w_pad=0.5)
plt.show()
fig.savefig('multi_raster_source_chips_normalized.png', bbox_inches='tight', pad_inches=0.)

#StatsTransformer_graph
fig, (ax_l, ax_r) = plt.subplots(1, 2, squeeze=True, figsize=(12, 5))

# left
for i in range(13):
    sns.kdeplot(chip_multi[..., i].flat, ax=ax_l, label=band_names[i])
ax_l.set_xscale('log')
ax_l.set_ylim((0, 0.01))
ax_l.set_xlabel('Pixel values')
ax_l.legend()
ax_l.set_title('Raw')

# right
for i in range(13):
    sns.kdeplot(
        chip_multi_normalized[..., i].flat, ax=ax_r, label=band_names[i])
ax_r.set_xlabel('Pixel values')
ax_r.legend()
ax_r.set_title('With StatsTransformer')

plt.show()
fig.savefig('stats_transformer_hists.png', bbox_inches='tight', pad_inches=0.2)

#consider_two_images
img_1_paths = sorted((img_dir / 'beirut' / 'imgs_1/').glob('*.tif'))
raster_sources_img_1 = [RasterioSource(uris=[str(path)]) for path in img_paths]

img_2_paths = sorted((img_dir / 'beirut' / 'imgs_2/').glob('*.tif'))
raster_sources_img_2 = [RasterioSource(uris=[str(path)]) for path in img_paths]

multi_rs_26 = MultiRasterSource(
    raster_sources=raster_sources_img_1 + raster_sources_img_2,
    primary_source_idx=1)

stats = RasterStats()
stats.compute(raster_sources=[multi_rs_26])

means_26=stats.means
stds_26=stats.means
stats_transformer = StatsTransformer(means_26,stds_26)

multi_rs_26_normalized = MultiRasterSource(
    raster_sources=raster_sources_img_1 + raster_sources_img_2,
    primary_source_idx=1, 
    raster_transformers=[stats_transformer])

#LabelSource
from rastervision.core.data import SemanticSegmentationLabelSource
from rastervision.core.data import ReclassTransformer

# the labels are just another raster, so we use the RasterioSource
label_raster_path = str(train_labels_dir / 'beirut/cm/beirut-cm.tif')
label_raster_source = RasterioSource(
    uris=[label_raster_path], 
    # make the class IDs start at zero by
    # remapping them using the ReclassTransformer
    raster_transformers=[ReclassTransformer({1: 0, 2: 1})])
class_config = [0, 1]
label_source = SemanticSegmentationLabelSource(raster_source=label_raster_source,class_config=class_config)

#400x400window
chip_label = label_source.get_labels(window=window)
chip_label_arr = chip_label.get_label_arr(window=window)

#labelled_map
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

def draw_label_chip(chip, ax):
    cmap = ListedColormap(sns.color_palette(n_colors=2))
    im = ax.imshow(chip, cmap=cmap)
    legend_items = [
        mpatches.Patch(facecolor=cmap(0), edgecolor='black', label='no change'),
        mpatches.Patch(facecolor=cmap(1), edgecolor='black', label='change')
    ]
    ax.legend(handles=legend_items, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 0))
    ax.axis('off')

fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(6, 6))
draw_label_chip(chip_label_arr, ax)
plt.show()
fig.savefig('label_source_chip.png', bbox_inches='tight', pad_inches=0.)

#Scenes
from rastervision.core.data import Scene
from rastervision.core.box import Box
import rasterio

scene = Scene(
    id='beirut',
    raster_source=multi_rs_26_normalized,
    label_source=label_source)


#GeoDataset
from rastervision.pytorch_learner import (
    SemanticSegmentationRandomWindowGeoDataset, GeoDataWindowMethod)

ds = SemanticSegmentationRandomWindowGeoDataset(
    scene=scene,
    size_lims=(80, 120),
    out_size=256,
    padding=100,
    max_windows=200)

#model
from torchvision import models
import torch.nn as nn

# Define the number of classes in your segmentation task
num_classes = 2  # Replace with the actual number of classes

# Load the pretrained Deeplabv3 model
model = models.segmentation.deeplabv3_resnet50(pretrained=True)

# Modify the first layer to accept 26-channel images
model.backbone.conv1 = nn.Conv2d(26, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Change the number of output classes to match your task
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

