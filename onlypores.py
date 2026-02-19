"""
Pore Detection and Material Segmentation Module

This module provides comprehensive functions for detecting pores in 3D volumes using various 
thresholding techniques. It includes functionality for Sauvola and Otsu thresholding, material 
masking, and pore extraction from X-ray CT scan data. The module supports both parallel and 
sequential processing to optimize performance based on available system memory.

Key Features:
- Adaptive Sauvola thresholding for variable background conditions
- Otsu thresholding for binary segmentation
- Material mask generation with void filling
- Pore cleaning and filtering based on size and dimensional constraints
- Memory-aware processing (automatic fallback to sequential processing)
- 3D morphological operations for noise reduction

Typical Workflow:
1. Load 3D X-ray CT volume
2. Apply onlypores() to extract pore structures
3. Optionally apply clean_pores() for additional filtering
4. Analyze resulting binary pore mask

Author: [Author name]
Date: [Date]
Version: 1.0
"""

import numpy as np
from skimage import measure
from skimage.measure import regionprops
from skimage import filters
import fill_voids
from joblib import Parallel, delayed
from skimage.filters import threshold_sauvola
from skimage.morphology import remove_small_objects
from skimage.measure import label
import psutil
from tqdm import tqdm
import logging


logger = logging.getLogger(__name__)

def sauvola_thresholding_concurrent(volume, window_size, k):
    """
    Apply Sauvola thresholding to a 3D volume using parallel processing along the Y-axis.
    
    Sauvola thresholding is an adaptive thresholding method that works well for images with varying
    background intensity. This function applies the algorithm slice by slice to a 3D volume using
    parallel processing for improved performance on multi-core systems.
    
    The algorithm computes a local threshold for each pixel based on the local mean and standard
    deviation within a specified window. This makes it particularly effective for volumes with
    uneven illumination or varying material densities.

    Parameters:
    ----------
    volume : numpy.ndarray
        3D numpy array representing the input volume to be thresholded.
        Expected shape: (Z, Y, X) where Z is depth, Y is height, X is width.
    window_size : int
        Size of the local window for Sauvola thresholding. Must be odd for proper centering.
        Larger values provide smoother thresholding but may miss fine details.
        Typical range: 15-101 pixels.
    k : float
        Parameter that controls threshold sensitivity. Higher values make thresholding more
        conservative (fewer pixels classified as features). Typical range: 0.05-0.5.

    Returns:
    -------
    numpy.ndarray
        3D binary numpy array after Sauvola thresholding with dtype=bool.
        True values represent material (above threshold), False represent pores/features.
        
    Notes:
    -----
    - Window size is automatically adjusted to be odd if an even number is provided
    - Processing is parallelized across Y-axis slices for optimal memory usage
    - The r parameter is fixed at 128 for standard dynamic range handling
    """
    # Ensure window size is odd for proper algorithm implementation
    # Even window sizes cause asymmetric neighborhoods around pixels
    if window_size % 2 == 0:
        window_size += 1
        logger.info(f"Window size adjusted to {window_size} (must be odd)")

    def sauvola_slice(slice_):
        """
        Apply Sauvola thresholding to a single 2D slice.
        
        This inner function is used by the parallel processing framework
        to apply thresholding to individual YZ slices of the volume.
        """
        # Compute adaptive threshold using local statistics
        sauvola_threshold = threshold_sauvola(slice_, window_size=window_size, k=k, r=128)
        # Values above threshold are considered material (True)
        # Values below threshold are considered pores/features (False)
        return slice_ > sauvola_threshold

    # Apply Sauvola thresholding to each YZ slice (volume[:, i, :]) in parallel
    # This approach minimizes memory usage while maximizing parallelization
    logger.info("Applying Sauvola thresholding with parallel processing...")
    binary_volume = np.array(Parallel(n_jobs=-1)(
        delayed(sauvola_slice)(volume[:, i, :]) for i in range(volume.shape[1])
    ))

    # Reshape result: parallel processing returns (Y, Z, X), we need (Z, Y, X)
    binary_volume = np.transpose(binary_volume, (1, 0, 2))

    return binary_volume

def sauvola_thresholding_nonconcurrent(volume, window_size, k):
    """
    Apply Sauvola thresholding to a 3D volume using sequential processing.
    
    This is a memory-efficient, sequential implementation of Sauvola thresholding that processes
    the volume slice by slice without parallel processing. Used when system memory is insufficient
    for the concurrent version or when debugging is needed.

    Parameters:
    ----------
    volume : numpy.ndarray
        3D numpy array representing the input volume to be thresholded.
        Expected shape: (Z, Y, X).
    window_size : int
        Size of the local window for Sauvola thresholding. Automatically adjusted to be odd.
    k : float
        Parameter that controls threshold sensitivity.

    Returns:
    -------
    numpy.ndarray
        3D binary numpy array after Sauvola thresholding with dtype=bool.
        
    Notes:
    -----
    - Lower memory footprint compared to concurrent version
    - Slower processing but more stable for large volumes
    - Suitable for systems with limited RAM
    """
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
        logger.info(f"Window size adjusted to {window_size} (must be odd)")
    
    # Pre-allocate output array for memory efficiency
    binary_volume = np.zeros_like(volume, dtype=bool)
    
    logger.info("[INFO] Applying Sauvola thresholding sequentially...")
    # Process each YZ slice sequentially
    for i in range(volume.shape[1]):
        slice_ = volume[:, i, :]
        # Compute adaptive threshold for current slice
        sauvola_threshold = threshold_sauvola(slice_, window_size=window_size, k=k, r=128)
        # Apply threshold and store result
        binary_volume[:, i, :] = slice_ > sauvola_threshold
        
        # Progress indicator for long operations
        if (i + 1) % 50 == 0:
            logger.info(f"[INFO] Processed {i + 1}/{volume.shape[1]} slices")
    
    return binary_volume

def sauvola_thresholding(volume, window_size, k):
    """
    Apply Sauvola thresholding to a 3D volume with automatic memory management.
    
    This is the main entry point for Sauvola thresholding. It automatically selects between
    parallel and sequential implementations based on available system memory to prevent
    memory overflow while maximizing performance.

    Parameters:
    ----------
    volume : numpy.ndarray
        3D numpy array representing the input volume to be thresholded.
    window_size : int
        Size of the local window for Sauvola thresholding.
    k : float
        Parameter that controls threshold sensitivity.

    Returns:
    -------
    numpy.ndarray
        3D binary numpy array after Sauvola thresholding.
        
    Notes:
    -----
    - Automatically chooses implementation based on memory availability
    - Memory estimation assumes ~2x input size for parallel processing overhead
    - Falls back to sequential processing if memory is insufficient
    """
    # Estimate memory requirements for parallel processing
    # Factor of 2 accounts for temporary arrays and processing overhead
    required_mem_bytes = volume.nbytes * 2
    available_mem_bytes = psutil.virtual_memory().available
    
    logger.info(f"Volume size: {volume.nbytes / 1024**3:.2f} GB")
    logger.info(f"Available memory: {available_mem_bytes / 1024**3:.2f} GB")
    logger.info(f"Required memory estimate: {required_mem_bytes / 1024**3:.2f} GB")
    
    # Choose implementation based on memory availability
    if available_mem_bytes > required_mem_bytes:
        logger.info("Using parallel implementation...")
        return sauvola_thresholding_concurrent(volume, window_size, k)
    else:
        logger.info("Insufficient memory for parallel processing. Using sequential implementation...")
        return sauvola_thresholding_nonconcurrent(volume, window_size, k)

def otsu_thresholding(volume):
    """
    Apply Otsu's automatic thresholding to a 3D volume.
    
    Otsu's method automatically determines an optimal threshold value by minimizing
    intra-class variance (or maximizing inter-class variance) in the image histogram.
    This method works best when the histogram has a clear bimodal distribution.

    Parameters:
    ----------
    volume : numpy.ndarray
        3D numpy array representing the input volume. Non-zero values are used
        for threshold computation.

    Returns:
    -------
    numpy.ndarray
        3D binary numpy array after Otsu thresholding with dtype=bool.
        Values above threshold are True (material), below are False (background/pores).
        
    Notes:
    -----
    - Works best with bimodal intensity distributions
    - Global threshold applied to entire volume
    - Threshold value is printed for debugging/analysis purposes
    """
    # Compute optimal threshold using Otsu's method
    # This analyzes the histogram to find the threshold that best separates two classes
    threshold_value = filters.threshold_otsu(volume)
    
    logger.info(f'Applying Otsu thresholding with value: {threshold_value}')

    # Apply threshold: values above threshold are considered material (True)
    binary = volume > threshold_value

    return binary

def slice_cleaning(img,min_size=2):
    """
    Clean a 2D binary image by removing small objects below a size threshold.
    
    This function removes connected components in a binary image that are smaller
    than the specified minimum size. It is useful for eliminating noise and small
    artifacts from segmented images.

    Parameters:
    ----------
    img : numpy.ndarray
        2D binary numpy array representing the input image.
    min_size : int, optional (default=2)
        Minimum size (in pixels) for connected components to be retained.
    Returns:
    -------
    numpy.ndarray
        2D binary numpy array representing the cleaned image.
    """
    
    # Remove small objects below the specified size threshold
    cleaned_img = remove_small_objects(img > 0, min_size=min_size, connectivity=1)
    
    # Convert back to binary: non-zero labels become True
    cleaned_binary = cleaned_img > 0
    
    return cleaned_binary

def onlypores(xct, frontwall=0, backwall=0, sauvola_radius=30, sauvola_k=0.125, min_size_filtering=-1):
    """
    Extract pores from a 3D X-ray CT volume using advanced thresholding and segmentation.
    
    This is the main function for pore detection. It combines Sauvola adaptive thresholding
    with material mask generation to accurately identify pore structures within a material
    sample. The function includes preprocessing steps like volume cropping, wall exclusion,
    and optional post-processing filtering.

    Parameters:
    ----------
    xct : numpy.ndarray
        3D numpy array representing the X-ray CT volume data.
        Expected shape: (Z, Y, X) where Z is the scan direction.
    frontwall : int, optional (default=0)
        Index of the front wall slice to exclude from analysis.
        Set to 0 to include all slices from the beginning.
    backwall : int, optional (default=0)  
        Index of the back wall slice to exclude from analysis.
        Set to 0 to include all slices to the end.
    sauvola_radius : int, optional (default=30)
        Radius (window size) for Sauvola thresholding. Larger values provide
        smoother thresholding but may miss fine details.
    sauvola_k : float, optional (default=0.125)
        Sensitivity parameter for Sauvola thresholding. Higher values make
        thresholding more conservative. Typical range: 0.05-0.5.
    min_size_filtering : int, optional (default=-1)
        Minimum size threshold for pore filtering. If > 0, applies clean_pores()
        function to remove small artifacts. Set to -1 to disable filtering.

    Returns:
    -------
    tuple of numpy.ndarray
        - onlypores : 3D binary array with detected pores (True = pore, False = material)
        - sample_mask : 3D binary array defining the material boundaries  
        - binary : 3D binary array from initial thresholding step
        
    Notes:
    -----
    - Automatically crops volume to non-zero region for efficiency
    - Applies adaptive thresholding optimized for varying material densities
    - Generates material mask using Otsu thresholding and void filling
    - Optional size-based filtering to remove noise and artifacts
    - Returns None values if input volume contains no data
    
    Example:
    -------
    >>> pores, mask, binary = onlypores(ct_volume, frontwall=10, backwall=200, 
    ...                                sauvola_radius=25, min_size_filtering=50)
    >>> print(f"Detected {np.sum(pores)} pore voxels")
    """
    logger.info('Starting pore detection analysis...')
    
    logger.info('Computing volume bounding box...')
    # Step 1: Find the bounding box WITHOUT loading all indices into memory    
    # --- PART A: Find Z limits (Slice-by-slice check) ---
    min_z, max_z = -1, -1
    
    # Find the first slice with data
    for i in range(xct.shape[0]):
        if np.any(xct[i] > 0):
            min_z = i
            break
            
    if min_z == -1:  # Handle empty volumes
        logger.error("No non-zero values found in the volume")
        return None, None, None
        
    # Find the last slice with data (scanning backwards)
    for i in range(xct.shape[0] - 1, -1, -1):
        if np.any(xct[i] > 0):
            max_z = i
            break

    # --- PART B: Find Y and X limits (Projection) ---
    # We collapse the 3D volume to a 2D mask to find Y/X boundaries.
    # We only check slices between min_z and max_z to save time.
    
    projection_mask = np.zeros(xct.shape[1:], dtype=bool)
    
    logger.info("Projecting volume to determine Y/X boundaries...")
    for i in tqdm(range(min_z, max_z + 1), desc="Projecting slices", position=1, leave=False):
        # Accumulate where data exists. logical_or is memory efficient here.
        # We use '|= ' which is the in-place bitwise OR operator
        if np.any(xct[i] > 0): # Tiny optimization: skip if slice is empty (unlikely inside z-range)
             projection_mask |= (xct[i] > 0)

    # Now we simply find the bounding box of the 2D projection
    # This is fast and uses negligible memory compared to 3D indices
    y_inds, x_inds = np.nonzero(projection_mask)
    
    min_y, max_y = np.min(y_inds), np.max(y_inds)
    min_x, max_x = np.min(x_inds), np.max(x_inds)

    # Step 2: Print results (Logic remains the same, calculation is done above)
    logger.info(f'Original volume shape: {xct.shape}')
    logger.info(f'Data bounding box: Z[{min_z}:{max_z}], Y[{min_y}:{max_y}], X[{min_x}:{max_x}]')
    
    # Add small margin around data for edge effects in filtering operations
    margin = 2  
    min_z = max(0, min_z - margin)
    min_y = max(0, min_y - margin) 
    min_x = max(0, min_x - margin)
    max_z = min(xct.shape[0] - 1, max_z + margin)
    max_y = min(xct.shape[1] - 1, max_y + margin)
    max_x = min(xct.shape[2] - 1, max_x + margin)

    logger.info(f'Cropped volume shape will be: Z[{min_z}:{max_z}], Y[{min_y}:{max_y}], X[{min_x}:{max_x}]')
    
    # Add small margin around data for edge effects in filtering operations
    # This prevents artifacts at volume boundaries during convolution operations
    margin = 2  
    min_z = max(0, min_z - margin)
    min_y = max(0, min_y - margin) 
    min_x = max(0, min_x - margin)
    max_z = min(xct.shape[0] - 1, max_z + margin)
    max_y = min(xct.shape[1] - 1, max_y + margin)
    max_x = min(xct.shape[2] - 1, max_x + margin)
    
    # Step 3: Extract the cropped volume for processing efficiency
    cropped_volume = xct[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
    logger.info(f'Cropped volume shape: {cropped_volume.shape}')
    
    # Step 4: Apply adaptive thresholding to detect material vs. background
    logger.info('Applying Sauvola adaptive thresholding...')
    binary_cropped = sauvola_thresholding(cropped_volume, window_size=sauvola_radius, k=sauvola_k)

    # Step 6: Handle wall exclusions for sample boundaries
    # Set wall regions to True (material) to exclude them from pore detection
    if frontwall > 0:
        logger.info(f'Excluding front wall: slices 0 to {frontwall-1}')
        binary_cropped[:frontwall, :, :] = True
    if backwall > 0:
        logger.info(f'Excluding back wall: slices {backwall} to end')
        binary_cropped[backwall:, :, :] = True
    
    # Step 7: Reconstruct full-size binary volume
    # Create binary volume matching original dimensions
    binary = np.zeros(xct.shape, dtype=bool)
    # Place processed data back into correct spatial location
    binary[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = binary_cropped
    
    # Step 8: Generate material mask to define sample boundaries
    logger.info('Generating material mask...')
    sample_mask_cropped = material_mask(cropped_volume)
    
    # Reconstruct full-size material mask
    sample_mask = np.zeros_like(binary)
    sample_mask[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1] = sample_mask_cropped
    
    # Step 9: Extract pores by combining thresholding and material mask
    # Invert binary: True becomes False (material -> background), False becomes True (pores -> foreground)
    binary_inverted = np.invert(binary)
    # Keep only pores that are within the material sample (intersection)
    onlypores_result = np.logical_and(binary_inverted, sample_mask)
    
    logger.info(f'Initial pore detection complete. Found {np.sum(onlypores_result)} pore voxels.')

    # Step 10: Optional post-processing to remove small artifacts
    if min_size_filtering > 0:
        logger.info(f'Applying pore filtering with minimum size: {min_size_filtering}')
        onlypores_result = clean_pores(onlypores_result, min_size=min_size_filtering)
        logger.info(f'After filtering: {np.sum(onlypores_result)} pore voxels remaining.')

    logger.info('Pore detection analysis complete.')
    return onlypores_result, sample_mask, binary

def material_mask(xct):
    """
    Generate a material mask for a 3D volume using sequential processing.
    
    This is a memory-efficient, sequential version of material mask generation.
    It applies the same algorithm as the parallel version but processes the entire
    volume at once without chunking. Used when memory is insufficient for parallel
    processing or for debugging purposes.

    Parameters:
    ----------
    xct : numpy.ndarray
        3D numpy array representing the input CT volume.

    Returns:
    -------
    numpy.ndarray
        3D binary numpy array representing the material mask where True indicates
        material regions and False indicates background/air.
        
    Notes:
    -----
    - Lower memory footprint than parallel version
    - Processes entire volume as single unit
    - Same algorithm: Otsu + max projection + void filling
    - More stable for very large volumes on memory-limited systems
    """
    logger.info('Computing material mask using sequential processing...')
    
    # Apply global Otsu thresholding to entire volume
    threshold_value = filters.threshold_otsu(xct)
    binary = xct > threshold_value
    
    # Generate maximum intensity projection along Z-axis
    max_proj = np.max(binary, axis=0)
    
    # Find connected components in the projection
    labels = measure.label(max_proj)
    props = regionprops(labels)
    
    if len(props) > 0:
        # Use bounding box of largest component (main sample)
        minr, minc, maxr, maxc = props[0].bbox
        
        # Crop binary volume to sample region
        binary_cropped = binary[:, minr:maxr, minc:maxc]
        
        # Fill internal voids to create solid material boundary
        logger.info('Filling internal voids...')
        sample_mask_cropped = fill_voids.fill(binary_cropped, in_place=False)
        
        # Reconstruct full-size mask
        sample_mask = np.zeros_like(binary)
        sample_mask[:, minr:maxr, minc:maxc] = sample_mask_cropped
    else:
        # Fallback: use thresholded volume directly
        logger.warn('No connected components found, using raw threshold')
        sample_mask = binary
    
    logger.info('Material mask generation complete.')
    return sample_mask
    
def clean_pores(onlypores, min_size=8):
    """
    Clean and filter detected pores by removing small artifacts and dimensionally constrained objects.
    
    This function performs post-processing on detected pores to remove noise and artifacts
    that may have been incorrectly identified as pores. It applies two main filtering criteria:
    1. Minimum volume threshold (removes small noise objects)
    2. Minimum dimensional extent (removes flat/thin artifacts)

    Parameters:
    ----------
    onlypores : numpy.ndarray
        3D binary numpy array with detected pores where True represents pore voxels
        and False represents material/background.
    min_size : int, optional (default=8)
        Minimum number of voxels for a connected component to be retained.
        Objects smaller than this threshold are considered noise and removed.

    Returns:
    -------
    numpy.ndarray
        3D binary numpy array with cleaned pores, explicitly cast to bool dtype.
        Only pores meeting both size and dimensional criteria are retained.
        
    Notes:
    -----
    - Uses 3D connectivity (26-neighborhood) for component labeling
    - Removes objects with less than 2 voxels extent in any spatial dimension
    - Dimensional filtering prevents retention of flat/linear artifacts
    - Essential for removing scanning artifacts and noise
    
    Algorithm Steps:
    1. Label connected components in 3D
    2. Remove components smaller than min_size voxels
    3. Analyze bounding box dimensions of remaining components
    4. Retain only components with ≥2 voxels in all axes (X, Y, Z)
    
    Example:
    -------
    >>> # Clean pores with minimum 50 voxels and 2-voxel dimensional extent
    >>> cleaned = clean_pores(detected_pores, min_size=50)
    >>> print(f"Removed {np.sum(detected_pores) - np.sum(cleaned)} noise voxels")
    """
    
    logger.info(f'Cleaning pores with min_size={min_size}...')
    
    # Step 1: Label connected components using 3D connectivity
    # Connectivity=3 means 26-neighborhood (face, edge, and corner neighbors)
    labeled_pores = label(onlypores, connectivity=3)
    initial_components = np.max(labeled_pores)
    logger.info(f'  Initial connected components: {initial_components}')

    # Step 2: Remove small objects based on voxel count
    # This eliminates single-voxel noise and very small artifacts
    labeled_pores = remove_small_objects(labeled_pores, min_size=min_size, connectivity=3)
    remaining_after_size = len(np.unique(labeled_pores)) - 1  # -1 for background
    logger.info(f'  Components after size filtering: {remaining_after_size}')

    # Step 3: Analyze dimensional properties of remaining components
    props = regionprops(labeled_pores)
    
    # Step 4: Apply dimensional filtering
    # Create output array for components meeting all criteria
    cleaned_pores = np.zeros_like(onlypores, dtype=bool)
    valid_components = 0

    pore_labels = []
    
    for prop in props:
        # Extract bounding box coordinates: (min_z, min_y, min_x, max_z, max_y, max_x)
        bbox = prop.bbox
        
        # Calculate spatial extent in each dimension
        z_extent = bbox[3] - bbox[0]  # max_z - min_z
        y_extent = bbox[4] - bbox[1]  # max_y - min_y  
        x_extent = bbox[5] - bbox[2]  # max_x - min_x
        
        # Apply dimensional criteria: must have ≥2 voxels in ALL dimensions
        # This prevents retention of flat planes or linear artifacts
        if z_extent >= 2 and y_extent >= 2 and x_extent >= 2:
            # Component meets all criteria - add to cleaned result
            pore_labels.append(prop.label)
            valid_components += 1

    logger.info(f'  Valid components after dimensional filtering: {valid_components}')

    #cleaning the pores based on valid labels
    cleaned_pores[np.isin(labeled_pores, pore_labels)] = True

    logger.info(f'  Components after dimensional filtering: {valid_components}')
    logger.info(f'  Total pore voxels retained: {np.sum(cleaned_pores)}')
    
    # Step 5: Ensure output is explicitly binary
    # This guarantees consistent dtype regardless of input variations
    return cleaned_pores.astype(bool)