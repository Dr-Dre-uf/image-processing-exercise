import streamlit as st
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float
import cv2  # Import OpenCV
from skimage import exposure
import os  # Import the os module

# --- Set page config for a modern look ---
st.set_page_config(
    page_title="Image Transformation Exercises",
    page_icon=":camera:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Images ---
@st.cache_data  # Cache the images to avoid reloading on every interaction
def load_images():
    IF_image_path = "assets/IFCells.jpg"
    mandrill_image = io.imread("http://sipi.usc.edu/database/download.php?vol=misc&img=4.2.03")
    IF = io.imread(IF_image_path)
    kidney_mri = io.imread("assets/kidney_mri.jpg")  # Load kidney_mri
    breast_img = cv2.imread('assets/breast.png', cv2.IMREAD_GRAYSCALE)  # Load breast image
    I_low_contrast = io.imread("assets/low_contrast2.jpg")

    # Ensure images have 3 channels (RGB)
    if IF.shape[-1] == 4:
        IF = IF[:, :, :3]
    if mandrill_image.shape[-1] == 4:
        mandrill_image = mandrill_image[:, :, :3]

    return IF, mandrill_image, kidney_mri, breast_img, I_low_contrast

IF, mandrill_image, kidney_mri, breast_img, I_low_contrast = load_images()

# --- Disclaimer ---
st.warning("This is a public application. Please do not upload any sensitive or private data.")

# --- Sidebar for Section Selection ---
st.sidebar.title("Section Selection")
selected_section = st.sidebar.radio(
    "Choose a Section:",
    (
        "Channel Separation",
        "Gamma Correction",
        "Contrast Adjustment",
        "Negative Transformation",
        "Histogram Equalization",
        "CLAHE",
        "Non-linear Intensity Transformation",
    )
)

# --- Image Selection ---
image_choice = st.sidebar.radio(
    "Select Image:",
    ("Mandrill", "Fluorescence", "Kidney MRI")
)

if image_choice == "Mandrill":
    img = mandrill_image
elif image_choice == "Fluorescence":
    img = IF
else:
    img = kidney_mri

# --- Image Upload ---
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = io.imread(uploaded_file)  # Load the uploaded image
    st.sidebar.success("Image uploaded successfully!")

# --- Section Content ---

if selected_section == "Channel Separation":
    st.header("Channel Separation")
    st.markdown("Visualizing Red, Green, and Blue Channels...")

    if img.ndim == 3:
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    else:
        st.warning("Please select an RGB image for channel separation.")
        R, G, B = 0, 0, 0

    channel_choice = st.selectbox(
        "Select Channel:",
        ("Red", "Green", "Blue")  # Updated channel names
    )

    if channel_choice == "Red":
        channel_image = R
    elif channel_choice == "Green":
        channel_image = G
    else:
        channel_image = B

    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    if channel_choice == "Red":
        ax.imshow(channel_image, cmap="Reds")
    elif channel_choice == "Green":
        ax.imshow(channel_image, cmap="Greens")
    else:
        ax.imshow(channel_image, cmap="Blues")
    ax.set_title(f"{channel_choice} Channel")
    ax.axis('off')
    st.pyplot(fig)

elif selected_section == "Gamma Correction":
    st.header("Gamma Correction")
    st.markdown("Adjusting the brightness of the image using a non-linear operation.")

    gamma_tooltip = "Adjusts the overall brightness of the image. Values less than 1 darken the image, while values greater than 1 brighten it."
    gamma = st.slider("Gamma Value", 0.1, 3.0, 1.0, help=gamma_tooltip)

    # Apply gamma correction
    corrected_image = np.uint8(np.clip(255 * (img / 255) ** gamma, 0, 255))

    # Display the original and corrected images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[1].imshow(corrected_image)
    ax[1].set_title(f"Gamma Corrected (γ = {gamma})")
    st.pyplot(fig)

elif selected_section == "Contrast Adjustment":
    st.header("Contrast Adjustment")
    st.markdown("Adjusting the contrast of the image.")

    min_tooltip = "Sets the lower bound of the intensity range used for rescaling."
    max_tooltip = "Sets the upper bound of the intensity range used for rescaling."

    in_range_min = st.slider("In Range Min", 0.0, 1.0, 0.55, help=min_tooltip)
    in_range_max = st.slider("In Range Max", 0.0, 1.0, 0.7, help=max_tooltip)

    # Apply contrast adjustment
    adj_img = exposure.rescale_intensity(img_as_float(img), in_range=(in_range_min, in_range_max), out_range=(0, 1))

    # Display the original and adjusted images
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[0].axis("off")
    ax[1].imshow(adj_img, cmap='gray')
    ax[1].set_title("Adjusted")
    ax[1].axis("off")

    fig.suptitle("Contrast Adjustment")
    plt.tight_layout()
    st.pyplot(fig)

elif selected_section == "Negative Transformation":
    st.header("Negative Transformation")
    st.markdown("Inverting the colors of the image.")

    # Interactive Negative Transformation
    inv_level_tooltip = "Adjusts the overall intensity of the inverted image."
    inv_level = st.slider("Inversion Level", 0, 255, 127, help=inv_level_tooltip)  # Slider for inversion level

    # Apply negative transformation based on the slider value
    inverted_image = 255 - (img * (255 / np.max(img))).astype(np.uint8)
    inverted_image = np.clip(inverted_image + inv_level, 0, 255).astype(np.uint8)

    # Display the original and negative images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)
    with col2:
        st.image(inverted_image, caption="Negative Image", use_container_width=True)

elif selected_section == "Histogram Equalization":
    st.header("Histogram Equalization")
    st.markdown("Image contrast enhancement using histogram analysis.")

    bins_tooltip = "Controls the number of bins used to create the histogram. Higher values provide more detail but may be noisier."
    num_bins = st.slider("Number of Histogram Bins", 10, 256, 256, help=bins_tooltip)

    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img

    fig, ax = plt.subplots(2, 2, figsize=(16, 8))
    ax[0, 0].imshow(img_gray, cmap='gray')
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Original Image')

    ax[0, 1].hist(img_gray.ravel(), bins=num_bins, color='blue')
    ax[0, 1].set_title('Original Histogram')
    ax[0, 1].set_ylim(0, 4000)

    # Apply histogram equalization
    img_eq = cv2.equalizeHist(img_gray)

    ax[1, 0].imshow(img_eq, cmap='gray')
    ax[1, 0].axis('off')
    ax[1, 0].set_title('Equalized Image')

    ax[1, 1].hist(img_eq.ravel(), bins=num_bins, color='blue')
    ax[1, 1].set_title('Equalized Histogram')
    ax[1, 1].set_ylim(0, 4000)

    fig.suptitle('Histogram Equalization')
    plt.tight_layout()
    st.pyplot(fig)

elif selected_section == "CLAHE":
    st.header("CLAHE")
    st.markdown("Image enhancement by applying Contrast Limited Adaptive Histogram Equalization (CLAHE).")

    clip_tooltip = "Limits the contrast enhancement in each tile to prevent noise amplification."
    tile_tooltip = "Determines the size of the tiles used for local histogram equalization. Smaller tiles provide more localized enhancement but can introduce artifacts."

    clip_limit = st.slider("Clip Limit", 1.0, 10.0, 2.0, help=clip_tooltip)  # Interactive clip limit
    tile_grid_size = st.slider("Tile Grid Size", 2, 32, 8, help=tile_tooltip)  # Interactive tile grid size

    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    clahe_img = clahe.apply(img_gray)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_gray, cmap='gray')
    ax[0].set_title("Original Image")
    ax[1].imshow(clahe_img, cmap='gray')
    ax[1].set_title("CLAHE Image")
    fig.suptitle("CLAHE Enhancement")
    plt.tight_layout()
    st.pyplot(fig)

elif selected_section == "Non-linear Intensity Transformation":
    st.header("Non-linear Intensity Transformation")
    st.markdown("Contrast enhancement of a low-contrast image using a non-linear intensity transformation.")

    # --- Tooltips using HTML and CSS ---
    m_tooltip = "Controls the shape of the intensity transformation curve. Higher values increase the contrast in darker regions."
    E_tooltip = "Determines the overall strength of the transformation. Higher values result in a more pronounced effect."

    m_val = st.slider("m Value", 0.0, 5.0, 0.5, help=m_tooltip)  # Interactive m value
    E_val = st.slider("E Value", 1.0, 100.0, 100.0, help=E_tooltip)  # Interactive E value

    G = 1. / (1. + m_val / (img_as_float(img) + 1e-4)) ** E_val

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img, cmap="gray")
    ax[0].axis("off")
    ax[0].set_title("Original Image")
    ax[1].imshow(G, cmap="gray")
    ax[1].axis("off")
    ax[1].set_title("Enhanced Contrast Image")
    fig.suptitle("Enhancing Contrast of Image")
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    st.pyplot(fig)