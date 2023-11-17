import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from sklearn.decomposition import PCA


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def show_image(list_img):
    for i, img in enumerate(list_img):
        plt.subplot(1, len(list_img), i+1)
        plt.imshow(img, cmap='gray')
    plt.show()


def extract_glcm_feature(img):
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    level = 256
    symmetric = True
    normed = True
    glcm = graycomatrix(img, distances, angles, level, symmetric, normed)

    df = pd.DataFrame()
    features = ['contrast', 'dissimilarity',
                'homogeneity', 'energy', 'correlation', 'ASM']
    angles = ['0', '45', '90', '135']

    properties = []
    columns = []
    for feature in features:
        feature_value = graycoprops(glcm, feature)
        for i, distance in enumerate(distances):
            for j, angle in enumerate(angles):
                properties.append(feature_value[i][j])
                columns.append(feature + '_' + str(distance) + '_' + angle)
    df = pd.DataFrame([properties], columns=columns)
    return df


def extract_lbp_feature(img):
    radius = [1, 2, 3]
    n_points = 8*radius
    METHOD = 'uniform'
    df = pd.DataFrame()
    for r in radius:
        columns = []
        n_points = 8*r
        lbp = local_binary_pattern(img, n_points, r, METHOD)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(
            lbp, density=True, bins=n_bins, range=(0, n_bins))
        hist = hist.reshape(-1, 1)
        # convert hist to vector
        hist = hist.flatten()
        for i in range(n_bins):
            columns.append('lbp_' + str(r) + '_' + str(i))
        df = pd.concat(
            [df, pd.DataFrame([hist], columns=columns)], axis=1)
    return df


def extract_gabor_feature(img):
    ksize = (5, 5)
    phi = 0

    df = pd.DataFrame()
    for theta in range(9):
        theta = theta * np.pi / 9
        for sigma in (1, 3):
            for lambd in (0.05, 0.25):
                for gamma in (0.05, 0.5):
                    kernel = cv2.getGaborKernel(
                        ksize, sigma, theta, lambd, gamma, phi, ktype=cv2.CV_32F)
                    image_gabor = ndi.convolve(img, kernel, mode='wrap')
                    mean = np.mean(image_gabor)
                    std = np.std(image_gabor)
                    properties = [mean, std]
                    theta_name =  int(theta / np.pi * 9 * 20)
                    columns = ['mean_' + str(theta_name) + '_' + str(sigma) + '_' + str(lambd) + '_' + str(gamma),
                            'std_' + str(theta_name) + '_' + str(sigma) + '_' + str(lambd) + '_' + str(gamma)]
                    df = pd.concat([df, pd.DataFrame([properties], columns=columns)], axis=1)

    return df


def main():
    path = 'colonies/'
    df = pd.DataFrame()
    # list_image = os.listdir(path)
    # name = list_image[1]
    # img = cv2.imread(path + name)
    # img = preprocess(img)
    # extract_lbp_feature(img, name.split('_')[0])
    for image_name in os.listdir(path):
        img = cv2.imread(path + image_name)
        img = preprocess(img)
        label = image_name.split('_')[0]
        feature_glcm = extract_glcm_feature(img)
        feature_lbp = extract_lbp_feature(img)
        feature_gabor = extract_gabor_feature(img)

        feature = pd.concat([feature_glcm, feature_lbp, feature_gabor], axis=1)
        feature['label'] = label
        df = pd.concat([df, feature], ignore_index=True)
    print(df)

def test():
    path = 'colonies/' + 'solitum_2.jpg'
    # list_image = os.listdir(path)
    # name = list_image[-1]
    img = cv2.imread(path)
    img = preprocess(img)
    extract_gabor_feature(img)


if __name__ == '__main__':
    main()
