# importing packages
import streamlit as st
from PIL import Image
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter, convolve
from scipy.signal import medfilt2d
import cv2
import datetime

# giving title for the page
st.set_page_config(page_title="FILTERAM: An app that filters your image", page_icon="random")

# a function that gives file name w.r.t date and time 
def generateFileName():
    currentDatetime = datetime.datetime.now()
    try:
        fileName = currentDatetime.strftime("%d_%b_%Y_%H_%M_%S.png")
        return fileName
    except ValueError:
        fileName = currentDatetime.strftime("%d_%b_%Y_%H_%M_%S.jpg")
        return fileName

# a function that saves the file
def saving(a, b, c):
    c1, c2, c3 = st.columns(3)
    with c2:
        button = st.button("Save", key=b)
        if button:
            try:
                # st.write(c+generateFileName())
                a.save(c+generateFileName())
                st.success("Image saved.")
            except AttributeError:
                cv2.imwrite(c+generateFileName(), a)
                st.success("Image saved.")


st.markdown(
    f"""
    <h3 style='color: #300030; font-family: Georgia;'>
        FILTERAM
    </h3>
    """
    , unsafe_allow_html=True
)




uploadedFile = st.sidebar.empty()
uploadedFile = st.sidebar.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])


if uploadedFile:
    st.write("YOUR SELECTED IMAGE.")
    uploadedImage = st.empty()
    uploadedImage.image(uploadedFile)
    

    image_1_1, image_1_2, image_1_3 = st.columns(3)

    with image_1_1:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Median Filter
    </h5>
    """, unsafe_allow_html=True)

        image = Image.open(uploadedFile)
        image_array = np.array(image)
        image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        imageGray = image.convert('L')
        imageArray = np.array(imageGray)


        filtered_image_array = median_filter(image_array, size=3)
        filtered_image = Image.fromarray(filtered_image_array)
        st.image(filtered_image, caption='Median Filtered')

        saving(filtered_image, "b1", "MedFil_")


        

    with image_1_2:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Gaussian Filter
    </h5>
    """, unsafe_allow_html=True)
        gaussianImageArray = gaussian_filter(image_array, sigma=1.5)
        gaussianFilteredImage = Image.fromarray(gaussianImageArray.astype(np.uint8))

        st.image(gaussianFilteredImage, caption='Gaussian Filtered')

        saving(gaussianFilteredImage, "b2", "GausFil_")


        
    with image_1_3:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Median Pass Filter
    </h5>
    """, unsafe_allow_html=True)

        passFiltered = medfilt2d(imageArray, kernel_size=3)
        passFilteredImage = Image.fromarray(passFiltered)

        st.image(passFilteredImage, caption='Median Passed')
        
        saving(passFilteredImage, "b3", "MedPass_")


    image_2_1, image_2_2, image_2_3 = st.columns(3)

    with image_2_1:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Low Pass Filter
    </h5>
    """, unsafe_allow_html=True)
        
        cutoff_frequency = 10
        sigma = 1.0 / (2 * np.pi * cutoff_frequency)
        lowPassimage_array = gaussian_filter(image_array, sigma=sigma)
        lowPassImage = Image.fromarray(lowPassimage_array.astype(np.uint8))

        st.image(lowPassImage, caption='Low Passed Image')

        saving(lowPassImage, "b4", "LowPass_")


    with image_2_2:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Linear Filter
    </h5>
    """, unsafe_allow_html=True)
        imageGray = image.convert('L')
        imageArray = np.array(imageGray)

        kernelLF = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        filtered_imageLF = cv2.filter2D(image_gray, -1, kernelLF)
        st.image(filtered_imageLF, caption='Linear Filtering')
    
        saving(filtered_imageLF, "b5", "LinFil_")


    with image_2_3:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        High Pass Filter
    </h5>
    """, unsafe_allow_html=True)

        blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
        high_pass = image_gray - blurred
        st.image(high_pass, caption='High-Pass Filtering')

        saving(high_pass, "b6", "HighPass_")


    image_3_1, image_3_2, image_3_3 = st.columns(3)

    with image_3_1:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Blurred
    </h5>
    """, unsafe_allow_html=True)
        image_array = np.array(image)

        normal_blur = cv2.blur(image_array, (5, 5))
        st.image(normal_blur, caption='Normal Blur')

        saving(normal_blur, "b7", "Blur_")


    with image_3_2:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Gaussian Blur
    </h5>
    """, unsafe_allow_html=True)
        gaussian_blur = cv2.GaussianBlur(image_array, (5, 5), 0)
        
        st.image(gaussian_blur, caption='Gaussian Blur')

        saving(gaussian_blur, "b8", "GausBlur_")


    with image_3_3:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Median Blur
    </h5>
    """, unsafe_allow_html=True)  
        
        median_blur = cv2.medianBlur(image_gray, 5)

        st.image(cv2.cvtColor(median_blur, cv2.COLOR_GRAY2RGB), caption='Median Blur') 

        saving(cv2.cvtColor(median_blur, cv2.COLOR_GRAY2RGB), "b9", "MedBlur_")     


    image_4_1, image_4_2, image_4_3 = st.columns(3)


    with image_4_1:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Log Transformed
    </h5>
    """, unsafe_allow_html=True)
        image_array = np.array(image)
        log_transformed = np.log1p(image_array)
        log_transformed = (log_transformed - np.min(log_transformed)) / (np.max(log_transformed) - np.min(log_transformed))
        log_transformed_image = Image.fromarray(np.uint8(log_transformed * 255))
        st.image(log_transformed_image, caption='Log Transformation')

        saving(log_transformed_image, "b10", "LogTr_")



    with image_4_2:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Power Transformed
    </h5>
    """, unsafe_allow_html=True)
        gamma = 0.5  # Power law gamma parameter
        power_law_image = np.power(image_array / 255.0, gamma)
        power_law_image = np.uint8(power_law_image * 255)
        st.image(power_law_image, caption='Power Law Transformation')

        saving(power_law_image, "b11", "Pow_")



    with image_4_3:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        DeSaturation
    </h5>
    """, unsafe_allow_html=True)
        desaturatedImage = image.convert('L')
        st.image(desaturatedImage, caption='DeSaturation')

        saving(desaturatedImage, "b12", "Desat_")


    image_5_1, image_5_2, image_5_3 = st.columns(3)

    with image_5_1:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Laplacian 
    </h5>
    """, unsafe_allow_html=True)
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        laplacian_edges = cv2.Laplacian(gray_image, cv2.CV_8U)
        st.image(laplacian_edges, caption='Laplacian Edge Detection')

        saving(laplacian_edges, "b13", "LapEd_")
    
    
    with image_5_2:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Histo Equal.
    </h5>
    """, unsafe_allow_html=True)
        hist_eq_image = cv2.equalizeHist(gray_image)
        st.image(hist_eq_image, caption='Histogram Equalization')

        saving(hist_eq_image, "b14", "HistEq_")
        
        

    with image_5_3:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Histo Adap Equal.
    </h5>
    """, unsafe_allow_html=True)
        adapt_eq_image = cv2.createCLAHE(clipLimit=2.0).apply(gray_image)
        st.image(adapt_eq_image, caption='Histogram Adaptive Equalization')

        saving(adapt_eq_image, "b15", "HistAdEq_")


    image_6_1, image_6_2, image_6_3 = st.columns(3)

    with image_6_1:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Sharpened alpha 1
    </h5>
    """, unsafe_allow_html=True)
        alpha = 1
        sharpened_image = cv2.Laplacian(image_array, cv2.CV_64F)
        scaled_image = alpha * sharpened_image
        sharpened_image = np.clip(image_array + scaled_image, 0, 255).astype(np.uint8)
        
        st.image(sharpened_image, caption=f'Sharpened Image (Alpha = {alpha})')

        saving(sharpened_image, "b16", "Sh1_")

    with image_6_2:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Sharpened alpha -5
    </h5>
    """, unsafe_allow_html=True)
        alpha = -5
        sharpened_image = cv2.Laplacian(image_array, cv2.CV_64F)
        scaled_image = alpha * sharpened_image
        sharpened_image = np.clip(image_array + scaled_image, 0, 255).astype(np.uint8)
        
        st.image(sharpened_image, caption=f'Sharpened Image (Alpha = {alpha})')

        saving(sharpened_image, "b17", "Sh2_")

    with image_6_3:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Sharpened alpha -1
    </h5>
    """, unsafe_allow_html=True)
        alpha = -1
        sharpened_image = cv2.Laplacian(image_array, cv2.CV_64F)
        scaled_image = alpha * sharpened_image
        sharpened_image = np.clip(image_array + scaled_image, 0, 255).astype(np.uint8)
        
        st.image(sharpened_image, caption=f'Sharpened Image (Alpha = {alpha})')

        saving(sharpened_image, "b18", "Sh3_")


    image_7_1, image_7_2, image_7_3 = st.columns(3)

    with image_7_1:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        White Top Hat
    </h5>
    """, unsafe_allow_html=True)
        gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        white_tophat = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)
        st.image(white_tophat, caption='White Top Hat')

        saving(white_tophat, "b19", "WHat_")

    with image_7_2:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Opening
    </h5>
    """, unsafe_allow_html=True)
        kernelO = np.ones((5, 5), np.uint8)
        openedImage = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernelO)
        st.image(openedImage, caption='Opening')

        saving(openedImage, "b20", "Op_")

    with image_7_3:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Black Top Hat
    </h5>
    """, unsafe_allow_html=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        black_tophat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
        st.image(black_tophat, caption='Black Top Hat')

        saving(black_tophat, "b21", "BHat_")

    image_8_1, image_8_2, image_8_3 = st.columns(3)

    with image_8_1:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Erosion
    </h5>
    """, unsafe_allow_html=True)

        grayImage = np.array(image.convert("L"))
        kernel_ = np.ones((5, 5), np.uint8)
        erodedImage = cv2.erode(grayImage, kernel_, iterations=1)
        st.image(erodedImage, caption='Erosion')

        saving(erodedImage, "b22", "Ero_")

    with image_8_2:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Dilation
    </h5>
    """, unsafe_allow_html=True)
        dilatedImage = cv2.dilate(grayImage, kernel_, iterations=1)
        st.image(dilatedImage, caption='Dilation')

        saving(dilatedImage, "b23", "Dil_")


    with image_8_3:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Closing
    </h5>
    """, unsafe_allow_html=True)
        closedImage = cv2.morphologyEx(grayImage, cv2.MORPH_CLOSE, kernel_)
        st.image(closedImage, caption='Closing')

        saving(closedImage, "b24", "Cl_")


    image_9_1, image_9_2, image_9_3 = st.columns(3)

    with image_9_1:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Decomposed 1
    </h5>
    """, unsafe_allow_html=True)
        r, g, b = image.split()
        st.image(r, caption='Decomposed 1')

        saving(r, "b25", "Dec1_")
        

    with image_9_2:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Decomposed 2
    </h5>
    """, unsafe_allow_html=True)
        st.image(g, caption='Decomposed 2')

        saving(g, "b26", "Dec2_")
    

    with image_9_3:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        Decomposed 3
    </h5>
    """, unsafe_allow_html=True)
        st.image(b, caption='Decomposed 3')

        saving(b, "b27", "Dec3_")
    
    image_10_1, image_10_2, image_10_3 = st.columns(3)

    with image_10_1:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        GrayScale: Only Green
    </h5>
    """, unsafe_allow_html=True)
        green_channel = image_array[:, :, 1]
        st.image(green_channel, caption='Grayscale with Green Channel')

        saving(green_channel, "b28", "Gr_")

    with image_10_2:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        GrayScale: Only Red
    </h5>
    """, unsafe_allow_html=True)
        red_channel = image_array[:, :, 0]
        st.image(red_channel, caption='Grayscale with Red Channel')

        saving(red_channel, "b29", "Rd_")

    with image_10_3:
        st.markdown("""
    <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
        GrayScale: Only Blue
    </h5>
    """, unsafe_allow_html=True)
        blue_channel = image_array[:, :, 2]
        st.image(blue_channel, caption='Grayscale with Blue Channel')

        saving(blue_channel, "b30", "Bl_")


    image_11_1, image_11_2, image_11_3 = st.columns(3)

    with image_11_1:
        st.markdown("""
            <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
                Contrast Stretching
            </h5>
            """, unsafe_allow_html=True)
        p_low, p_high = np.percentile(image, (5, 95))
        contrast_stretched = np.clip((image - p_low) * (255.0 / (p_high - p_low)), 0, 255).astype(np.uint8)
        st.image(contrast_stretched, caption='Contrast Stretching')


        saving(contrast_stretched, "b31", "ConStr_")

    with image_11_2:
        st.markdown("""
            <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
                Auto Brightness
            </h5>
            """, unsafe_allow_html=True)
        equalized_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        equalized_image[:, :, 0] = cv2.equalizeHist(equalized_image[:, :, 0])
        equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_LAB2RGB)
        st.image(equalized_image, caption='Auto Brightness')

        saving(equalized_image, "b32", "AutBr_")


    with image_11_3:
        st.markdown("""
            <h5 style='color: #300030; font-family: Helvetica; font-style: italic; font-weight: bold;'>
                Brightness Adjustment
            </h5>
            """, unsafe_allow_html=True)
        blurred = cv2.GaussianBlur(image_array, (0, 0), 3)
        unsharp_mask = cv2.addWeighted(image_array, 1.5, blurred, -0.5, 0)
        st.image(unsharp_mask, caption='High Definition Filter')

        saving(unsharp_mask, "b33", "HD_")

else:
    st.write("PLEASE SELECT AN IMAGE")


