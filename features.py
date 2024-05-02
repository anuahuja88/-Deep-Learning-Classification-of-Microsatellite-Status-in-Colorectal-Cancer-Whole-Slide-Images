import cv2
import os
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import skimage.feature as feature

def get_HSV_histogram(tile):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    
    # Calculate histogram
    hist_hue = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    hist_saturation = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    hist_value = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
    
    return hist_hue, hist_saturation, hist_value

def get_BGR_histogram(tile):

    B = cv2.calcHist([tile], [0], None, [256], [0, 256])
    G = cv2.calcHist([tile], [1], None, [256], [0, 256])
    R = cv2.calcHist([tile], [2], None, [256], [0, 256])
    
    return B, G, R

def extract_colour_histograms(true, false):
    mss_channel1_hist = []
    mss_channel2_hist = []
    mss_channel3_hist = []
    msi_channel1_hist = []
    msi_channel2_hist = []
    msi_channel3_hist = []
    # true tiles
    for tile in os.listdir(true):
        tile = cv2.imread(f"{true}/{tile}")
        # cv2.imshow("True tile",tile)
        # cv2.waitKey(0)
        true_channel1, true_channel2, true_channel3 = get_HSV_histogram(tile)
        msi_channel1_hist.append(true_channel1)
        msi_channel2_hist.append(true_channel2)
        msi_channel3_hist.append(true_channel3)
    # false tiles
    for tile in os.listdir(false):
        tile = cv2.imread(f"{false}/{tile}")
        # cv2.imshow("False tile",tile)
        # cv2.waitKey(0)
        false_channel1, false_channel2, false_channel3 = get_HSV_histogram(tile)
        mss_channel1_hist.append(false_channel1)
        mss_channel2_hist.append(false_channel2)
        mss_channel3_hist.append(false_channel3)


    # Combine histograms
    true_hist1 = np.mean(msi_channel1_hist, axis=0)
    true_hist2 = np.mean(msi_channel2_hist, axis=0)
    true_hist3 = np.mean(msi_channel3_hist, axis=0)

    false_hist1 = np.mean(mss_channel1_hist, axis=0)
    false_hist2 = np.mean(mss_channel2_hist, axis=0)
    false_hist3 = np.mean(mss_channel3_hist, axis=0)


    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(true_hist1, color='r')
    plt.title('H Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.subplot(1, 3, 2)
    plt.plot(true_hist2, color='g')
    plt.title('S Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.subplot(1, 3, 3)
    plt.plot(true_hist3, color='b')
    plt.title('V Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(false_hist1, color='r')
    plt.title('H Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.subplot(1, 3, 2)
    plt.plot(false_hist2, color='g')
    plt.title('S Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.subplot(1, 3, 3)
    plt.plot(false_hist3, color='b')
    plt.title('V Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def extract_texture(true, false):
    true_contrast_avg = []
    true_dissimilarity_avg = []
    true_homogeneity_avg = []
    true_energy_avg = []
    true_correlation_avg = []
    true_ASM_avg = []

    false_contrast_avg = []
    false_dissimilarity_avg = []
    false_homogeneity_avg = []
    false_energy_avg = []
    false_correlation_avg = []
    false_ASM_avg = []

    # True tiles
    for tile in os.listdir(true):
        tile = cv2.imread(f"{true}/{tile}")
        tile_grey = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

        # Calculate GLCM
        graycom = feature.graycomatrix(tile_grey, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

        # Find GLCM properties
        contrast = feature.graycoprops(graycom, 'contrast')
        dissimilarity = feature.graycoprops(graycom, 'dissimilarity')
        homogeneity = feature.graycoprops(graycom, 'homogeneity')
        energy = feature.graycoprops(graycom, 'energy')
        correlation = feature.graycoprops(graycom, 'correlation')
        ASM = feature.graycoprops(graycom, 'ASM')

        # Append GLCM properties to lists
        true_contrast_avg.append(contrast)
        true_dissimilarity_avg.append(dissimilarity)
        true_homogeneity_avg.append(homogeneity)
        true_energy_avg.append(energy)
        true_correlation_avg.append(correlation)
        true_ASM_avg.append(ASM)

    # False tiles
    for tile in os.listdir(false):
        tile = cv2.imread(f"{false}/{tile}")
        tile_grey = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

        # Calculate GLCM
        graycom = feature.graycomatrix(tile_grey, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

        # Find GLCM properties
        contrast = feature.graycoprops(graycom, 'contrast')
        dissimilarity = feature.graycoprops(graycom, 'dissimilarity')
        homogeneity = feature.graycoprops(graycom, 'homogeneity')
        energy = feature.graycoprops(graycom, 'energy')
        correlation = feature.graycoprops(graycom, 'correlation')
        ASM = feature.graycoprops(graycom, 'ASM')

        # Append GLCM properties to lists
        false_contrast_avg.append(contrast)
        false_dissimilarity_avg.append(dissimilarity)
        false_homogeneity_avg.append(homogeneity)
        false_energy_avg.append(energy)
        false_correlation_avg.append(correlation)
        false_ASM_avg.append(ASM)

    # Calculate average GLCM properties for true and false tiles
    true_contrast_avg = np.mean(true_contrast_avg)
    true_dissimilarity_avg = np.mean(true_dissimilarity_avg)
    true_homogeneity_avg = np.mean(true_homogeneity_avg)
    true_energy_avg = np.mean(true_energy_avg)
    true_correlation_avg = np.mean(true_correlation_avg)
    true_ASM_avg = np.mean(true_ASM_avg)

    false_contrast_avg = np.mean(false_contrast_avg)
    false_dissimilarity_avg = np.mean(false_dissimilarity_avg)
    false_homogeneity_avg = np.mean(false_homogeneity_avg)
    false_energy_avg = np.mean(false_energy_avg)
    false_correlation_avg = np.mean(false_correlation_avg)
    false_ASM_avg = np.mean(false_ASM_avg)

    # Plot individual graphs for each GLCM property
    properties = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM']
    true_averages = [true_contrast_avg, true_dissimilarity_avg, true_homogeneity_avg, true_energy_avg, true_correlation_avg, true_ASM_avg]
    false_averages = [false_contrast_avg, false_dissimilarity_avg, false_homogeneity_avg, false_energy_avg, false_correlation_avg, false_ASM_avg]

    for prop, true_avg, false_avg in zip(properties, true_averages, false_averages):
        plt.figure()
        plt.bar(['True', 'False'], [true_avg, false_avg], color=['blue', 'red'])
        plt.title(prop)
        plt.ylabel('Average Value')
        plt.show()

def local_binary_pattern(true, false):
    lbp_features_true = []
    lbp_features_false = []

    for tile in os.listdir(true):
        tile = cv2.imread(f"{true}/{tile}")
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 59))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)  # Avoid divide by zero

        # Append LBP features to the list
        lbp_features_true.append(hist)

    for tile in os.listdir(false):
        tile = cv2.imread(f"{false}/{tile}")
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 59))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)  # Avoid divide by zero

        # Append LBP features to the list
        lbp_features_false.append(hist)


    plt.figure(figsize=(10, 6))
    plt.bar(range(len(lbp_features_true[0])), np.mean(lbp_features_true, axis=0), color='b')
    plt.title("True LBP Features")
    plt.xlabel('LBP Pattern')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(lbp_features_false[0])), np.mean(lbp_features_false, axis=0), color='r')
    plt.title("False LBP Features")
    plt.xlabel('LBP Pattern')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()






def main():
    directory_true = "TRUE_Group_for_Microsatellite_Instability_dMMR_MSI/Tiles"
    directory_false = "FALSE_Group_for_Microsatellite_Instability_pMMR_MSS/Tiles"
    # extract_colour_histograms(directory_true, directory_false)
    # extract_texture(directory_true, directory_false)
    local_binary_pattern(directory_true, directory_false)
    

        

if __name__ == "__main__":
    main()