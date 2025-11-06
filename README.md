# hest_1k_data_analysis

### Inspect freshly downloaded samples

For each sample, we provide:

- **wsis/**: H&E-stained whole slide images in pyramidal Generic TIFF (or pyramidal Generic BigTIFF if >4.1GB)
- **st/**: Spatial transcriptomics expressions in a scanpy .h5ad object
- **metadata/**: Metadata
- **spatial_plots/**: Overlay of the WSI with the st spots
- **thumbnails/**: Downscaled version of the WSI
- **tissue_seg/**: Tissue segmentation masks:
    - `{id}_mask.jpg`: Downscaled or full resolution greyscale tissue mask
    - `{id}_mask.pkl`: Tissue/holes contours in a pickle file
    - `{id}_vis.jpg`: Visualization of the tissue mask on the downscaled WSI
- **pixel_size_vis/**: Visualization of the pixel size
- **patches/**: 256x256 H&E patches (0.5µm/px) extracted around ST spots in a .h5 object optimized for deep-learning. Each patch is matched to the corresponding ST profile (see **st/**) with a barcode.
- **patches_vis/**: Visualization of the mask and patches on a downscaled WSI.
- **transcripts/**: individual transcripts aligned to H&E for xenium samples; read with pandas.read_parquet; aligned coordinates in pixel are in columns `['he_x', 'he_y']`
- **cellvit_seg/**: Cellvit nuclei segmentation
- **xenium_seg**: xenium segmentation on DAPI and aligned to H&E
