from sklearn.cluster import KMeans
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
from itertools import product
import pyfeats
from glrlm import glrlm_features


def preprocess(img):
    # print(img.shape)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def show_image(list_img):
    for i, img in enumerate(list_img):
        plt.subplot(1, len(list_img), i+1)
        plt.imshow(img, cmap='gray')
    plt.show()


def extract_glcm_feature(img):
    distances = [3, 5]
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
    radius = [2, 3]
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
    ksize = (9, 9)
    sigma_s = [3]
    theta_s = [angle * np.pi / 9 for angle in range(9)]
    lambd_s = [1, 3]
    gamma_s = [0.05]
    phi_s = [0]

    df = pd.DataFrame()
    for theta, sigma, lambd, gamma, phi in product(theta_s, sigma_s, lambd_s, gamma_s, phi_s):
        kernel = cv2.getGaborKernel(
            ksize, sigma, theta, lambd, gamma, phi, ktype=cv2.CV_32F)
        image_gabor = ndi.convolve(img, kernel, mode='wrap')

        theta = theta / np.pi * 180
        phi = phi / np.pi * 180

        title_kernel = 'kernel' + \
            '_sigma' + str(sigma) + \
            '_theta' + str(theta) + \
            '_lambd' + str(lambd) + \
            '_gamma' + str(gamma) + \
            '_phi' + str(phi)
        # fig, axes = plt.subplots(1, 3)
        # list_image = [img, kernel, image_gabor]
        # title = ['original', title_kernel, 'gabor']
        # for i, ax in enumerate(axes):
        #     ax.imshow(list_image[i], cmap='gray')
        #     ax.set_title(title[i])
        # # set size figure
        # fig.set_size_inches(18.5, 10.5)
        # plt.show()
        mean = np.mean(image_gabor)
        std = np.std(image_gabor)
        properties = [mean, std]
        columns = ['mean_' + title_kernel, 'std_' + title_kernel]
        df = pd.concat(
            [df, pd.DataFrame([properties], columns=columns)], axis=1)

    return df


def extract_center(image):
    height, width, _ = image.shape
    center = (int(width/2), int(height/2))
    radius = int(width/2*0.3)
    mask = np.zeros_like(image)
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
    center_image = cv2.bitwise_and(image, mask)
    return center_image


def extract_middle(image):
    height, width, _ = image.shape
    mask_inner = np.zeros_like(image)
    center = (int(width/2), int(height/2))
    inner_radius = int(width/2*0.3)

    cv2.circle(mask_inner, center, inner_radius, (255, 255, 255), thickness=-1)

    # Tạo mặt nạ hình tròn ngoài (mặt nạ 2)
    outer_radius = int(width/2*0.8)
    mask_outer = np.zeros_like(image)
    cv2.circle(mask_outer, center, outer_radius, (255, 255, 255), thickness=-1)

    # Lấy khoảng viền ở mặt nạ ngoài không giao với mặt nạ trong
    outer_border = cv2.bitwise_xor(mask_outer, mask_inner)

    # Áp dụng mặt nạ để chọn phần ảnh trong khoảng viền
    middle_image = cv2.bitwise_and(image, outer_border)

    return middle_image


def extract_border(image):
    # Đọc ảnh từ đường dẫn
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    # Tạo mặt nạ hình tròn trong
    mask_inner = np.zeros_like(image)
    center = (int(width/2), int(height/2))
    inner_radius = int(width/2*0.8)

    cv2.circle(mask_inner, center, inner_radius, (255, 255, 255), thickness=-1)

    # Tạo mặt nạ hình tròn ngoài (mặt nạ 2)
    outer_radius = int(width/2)
    mask_outer = np.zeros_like(image)
    cv2.circle(mask_outer, center, outer_radius, (255, 255, 255), thickness=-1)

    # Lấy khoảng viền ở mặt nạ ngoài không giao với mặt nạ trong
    outer_border = cv2.bitwise_xor(mask_outer, mask_inner)

    # Áp dụng mặt nạ để chọn phần ảnh trong khoảng viền
    outer_border_image = cv2.bitwise_and(image, outer_border)
    return outer_border_image


def extract_color(image, region):
    features = []
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r_mean, g_mean, b_mean = np.mean(image, axis=(0, 1))
    r_std, g_std, b_std = np.std(image, axis=(0, 1))
    # Chuyển đổi ảnh sang không gian màu HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Trích xuất các thành phần màu sắc
    hue = hsv_image[:, :, 0]
    saturation = hsv_image[:, :, 1]
    value = hsv_image[:, :, 2]
    # Tính các thống kê đơn giản của mỗi thành phần màu
    hue_mean = np.mean(hue)
    saturation_mean = np.mean(saturation)
    value_mean = np.mean(value)
    hue_std = np.std(hue)
    saturation_std = np.std(saturation)
    value_std = np.std(value)

    # Đọc ảnh ảnh xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Tính histogram của ảnh xám
    hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    # Xác định giá trị có tần suất xuất hiện nhiều nhất (bin)
    dominant_gray_bin = np.argmax(hist_gray)
    features = [r_mean, r_std, g_mean, g_std, b_mean, b_std,
                # hue_mean, hue_std, saturation_mean, saturation_std, value_mean, value_std,
                dominant_gray_bin]
    columns = []
    for feature in ['r', 'g', 'b',
                    # 'hue', 'saturation', 'value'
                    ]:
        columns.append(feature + '_' + 'mean_' + region)
        columns.append(feature + '_' + 'std_' + region)
    columns += ['dominant_gray_bin_' + region]
    features = pd.DataFrame([features], columns=columns)
    return features

    # d={'avgR1':[],'avgG1':[],'avgB1':[],'avgR2':[],'avgG2':[],'avgB2':[],'avgR3':[],'avgG3':[],'avgB3':[],
    #    'avgH1':[],'avgS1':[],'avgV1':[],'avgH2':[],'avgS2':[],'avgV2':[],'avgH3':[],'avgS3':[],'avgV3':[],
    #    'stdR1':[],'stdG1':[],'stdB1':[],'stdR2':[],'stdG2':[],'stdB2':[],'stdR3':[],'stdG3':[],'stdB3':[],
    #    'stdH1':[],'stdS1':[],'stdV1':[],'stdH2':[],'stdS2':[],'stdV2':[],'stdH3':[],'stdS3':[],'stdV3':[],
    #    }


def extract_color_feature(image):
    center = extract_center(image)
    middle = extract_middle(image)
    border = extract_border(image)
    feature_color_center = extract_color(center, 'center')
    feature_color_middle = extract_color(middle, 'middle')
    feature_color_border = extract_color(border, 'border')
    feature = pd.concat(
        [feature_color_center, feature_color_middle, feature_color_border], axis=1)
    return feature


def extract_glrlm_feature(img):
    mask = np.ones_like(img)
    features, labels = glrlm_features(img, mask)
    feature = pd.DataFrame([features], columns=labels)
    return feature


def extract_glszm_feature(img):
    mask = np.ones_like(img)
    features, labels = pyfeats.glszm_features(img, mask)
    feature = pd.DataFrame([features], columns=labels)
    return feature


def extract_law_feature(img):
    mask = np.ones_like(img)
    features, labels = pyfeats.lte_measures(img, mask)
    feature = pd.DataFrame([features], columns=labels)
    return feature


def main():
    envs = ['MEA', 'YES']
    feature_name = 'glszm'
    for env in envs:
        path = f'colony/{env}/'
        df = pd.DataFrame()
        for image_name in os.listdir(path):
            img = cv2.imread(path + image_name)
            # feature_color = extract_color_feature(img)
            img = preprocess(img)
            label = image_name.split('_')[0]
            # feature_glcm = extract_glcm_feature(img)
            # feature_lbp = extract_lbp_feature(img)
            # feature_gabor = extract_gabor_feature(img)
            # feature_glrlm = extract_glrlm_feature(img)
            feature_glszm = extract_glszm_feature(img)
            # feature_law = extract_law_feature(img)
            feature = pd.concat([feature_glszm], axis=1)
            feature['label'] = label
            df = pd.concat([df, feature], ignore_index=True)
        print(df)
        df.to_csv(
            f'feature2/{feature_name}/feature_{feature_name}_{env}.csv', index=False)

    df = pd.DataFrame()
    for env in envs:
        df = pd.concat(
            [df, pd.read_csv(f'feature2/{feature_name}/feature_{feature_name}_{env}.csv')], ignore_index=True)
    df.to_csv(
        f'feature2/{feature_name}/feature_{feature_name}.csv', index=False)


def combine_feature():
    envs = ['CYA', 'DG18', 'MEA', 'YES']
    feature_names = ['color', 'glcm', 'lbp', 'gabor']
    df = pd.DataFrame()
    for env in envs:
        df_env = pd.DataFrame()
        for feature_name in feature_names:
            df_feature = pd.read_csv(
                f'feature2/{feature_name}/feature_{feature_name}_{env}.csv')
            df_env = pd.concat([df_env, df_feature], axis=1)
        # chỉ giữ lại 1 cột label
        df_env = df_env.loc[:, ~df_env.columns.duplicated(keep='last')]
        df_env.to_csv(f'feature2/full/feature_full_{env}.csv', index=False)
        df = pd.concat([df, df_env], ignore_index=True)
    df.to_csv(f'feature2/full/feature_full.csv', index=False)


def test():
    path = 'colonies/' + 'entromorphobium_0.jpg'

    # list_image = os.listdir(path)
    # name = list_image[-1]
    img = cv2.imread(path)
    img = preprocess(img)

    # extract_gabor_feature(img)
    df = extract_glszm_feature(img)
    print(df)


if __name__ == '__main__':
    main()
    # test()
    # combine_feature()
