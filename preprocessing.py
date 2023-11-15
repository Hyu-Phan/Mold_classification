from gettext import find
from re import S
from matplotlib.pylab import f
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os

import cv2
from matplotlib.pyplot import show
import numpy as np
from sklearn.cluster import KMeans


def get_path(path, id=0):
    if os.path.isfile(path):
        return path
    return get_path(path + '/' + os.listdir(path)[id])


def show_image(img, width=800):
    resized_img = cv2.resize(img, (1200,  800))

    while True:
        cv2.imshow('image', resized_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def determine_line(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        x1 += 1
    a = (y2 - y1) / (x2 - x1)  # Hệ số góc của đường thẳng
    b = y1 - a * x1  # Sai số của đường thẳng
    return a, b


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def draw_point(img, point):
    x, y = point
    cv2.circle(img, (x, y), 1, (0, 0, 255), 3)
    return img


def draw_circle(img):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=150, maxRadius=190)
    img = cv2.imread(path)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x,  y, r in circles[0, :]:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
    return img


def draw_contour(img, contour):
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)


def determine_petri_dish(img):
    # tính histogram

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = np.squeeze(hist)
    smooth_hist = gaussian_filter(hist, sigma=2)
    peeks, _ = find_peaks(-smooth_hist, prominence=5)
    threshold_value = peeks[0]+20
    img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)[1]
    img = cv2.Canny(img, 90, 200)
    show_image(img)
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour


def kmeans_clustering(points, n_clusters=2):
    points = np.array(points)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(points)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    return centers, labels


def determine_center_petri_dish(img):

    height, width = img.shape

    a1, b1 = determine_line((0, 0), (width, height))
    a2, b2 = determine_line((0, height), (width, 0))

    list_poins_contour = determine_petri_dish(img)
    draw_contour(img, list_poins_contour)

    center_point, radius = cv2.minEnclosingCircle(list_poins_contour)

    # làm tròn center_point
    center_point = (int(center_point[0]), int(center_point[1]))
    img = cv2.circle(img, center_point, int(radius), (0, 0, 255), 2)

    intersection_point = [[], []]
    for i in range(len(list_poins_contour)):
        x, y = list_poins_contour[i][0]
        if abs(y - (a1 * x + b1)) <= 25:
            intersection_point[0].append((x, y))
        if abs(y - (a2 * x + b2)) <= 25:
            intersection_point[1].append((x, y))

    print(intersection_point)
    intersection_point[0], _ = kmeans_clustering(intersection_point[0])
    intersection_point[1], _ = kmeans_clustering(intersection_point[1])

    print("Giao điểm:", np.around(intersection_point))
    a3, b3 = determine_line(intersection_point[0][0], intersection_point[0][1])
    a4, b4 = determine_line(intersection_point[1][0], intersection_point[1][1])

    center_point_line_1 = np.mean(intersection_point[0], axis=0)
    center_point_line_2 = np.mean(intersection_point[1], axis=0)

    # tìm đường thẳng vuông góc với a3, b3 và đi qua center_point_line_1
    a5 = -1 / a3
    b5 = center_point_line_1[1] - a5 * center_point_line_1[0]
    # tìm đường thẳng vuông góc với a4, b4 và đi qua center_point_line_2
    a6 = -1 / a4
    b6 = center_point_line_2[1] - a6 * center_point_line_2[0]

    # tìm giao điểm của 2 đường thẳng trên
    x = (b6 - b5) / (a5 - a6)
    y = a5 * x + b5
    # làm tròn thành số nguyên
    x = int(np.around(x))
    y = int(np.around(y))
    print("Vị trí tâm:", x, y)

    # tính bán kính của petri dish
    radius = np.linalg.norm(
        np.array(list_poins_contour[1][0]) - np.array((x, y)))
    radius = int(np.around(radius) - radius//10) 
    print("Bán kính:", radius)
    img = cv2.circle(img, (x, y), int(radius), (0, 255, 0), 2)
    show_image(img)
    # img = draw_point(img, (x, y))
    # img = cv2.circle(img, (x, y), int(radius), (0, 255, 0), 1)

    # show_image(img)
    return (x, y), radius


def determine_colony(points):
    (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
    s1 = x1**2 + y1**2
    s2 = x2**2 + y2**2
    s3 = x3**2 + y3**2
    M11 = x1*y2 + x2*y3 + x3*y1 - (x2*y1 + x3*y2 + x1*y3)
    M12 = s1*y2 + s2*y3 + s3*y1 - (s2*y1 + s3*y2 + s1*y3)
    M13 = s1*x2 + s2*x3 + s3*x1 - (s2*x1 + s3*x2 + s1*x3)
    x0 = int(0.5*M12 / M11)
    y0 = int(-0.5*M13 / M11)
    r0 = int(((x1 - x0)**2 + (y1 - y0)**2)**0.5)
    return (x0, y0), r0


def circle_intersection(circle1, circle2):
    (x1, y1), r1 = circle1
    (x2, y2), r2 = circle2
    d = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    if (d > r1 + r2) or \
        (d < abs(r1 - r2)) or \
            (d == 0 and r1 == r2):
        return None, None

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = (r1**2 - a**2)**0.5
    x0 = x1 + a * (x2 - x1) / d
    y0 = y1 + a * (y2 - y1) / d
    x3 = x0 + h * (y2 - y1) / d
    y3 = y0 - h * (x2 - x1) / d
    x4 = x0 - h * (y2 - y1) / d
    y4 = y0 + h * (x2 - x1) / d
    return (int(x3), int(y3)), (int(x4), int(y4))


def extract_colony(center, radius):
    img = cv2.imread(path)
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, (255, 255, 255), -1, 8, 0)
    masked_img = cv2.bitwise_and(img, mask)

    # cắt ảnh theo hình chữ nhật bao quanh hình tròn
    x, y = center
    masked_img = masked_img[y-radius:y+radius, x-radius:x+radius]

    show_image(masked_img)


def histogram_distance(img):
    center, radius = determine_center_petri_dish(img)

    distance = []
    points_of_distance = []
    for angle in range(360):
        points_of_radius = []
        intensity = []
        for i in range(radius, 0, -1):
            x = int(center[0] + i * np.cos(angle * np.pi / 180))
            y = int(center[1] + i * np.sin(angle * np.pi / 180))
            points_of_radius.append((x, y))
            intensity.append(img[y, x])

        gaussian_intensity = gaussian_filter(intensity, sigma=5)
        smooth_intensity = np.convolve(intensity, [1/3, 1/3, 1/3], mode='same')
        derivative = np.gradient(smooth_intensity)

        segments = []
        start_index = None
        segment_start = None
        for i in range(len(derivative)):
            if derivative[i] > 2.:
                if segment_start is None:
                    segment_start = i
            elif segment_start is not None:
                # segments.append((segment_start, i))
                long = i - segment_start
                if long > 5:
                    start_index = segment_start
                    break
                segment_start = None
        if start_index is None:
            start_index = radius-1
        #     peaks, _ = find_peaks(gaussian_intensity, prominence=10)
        #     plt.plot(gaussian_intensity)
        #     plt.plot(peaks, gaussian_intensity[peaks], "x")
        #     plt.show()
        img = draw_point(img, points_of_radius[start_index])
        distance.append(start_index)
        points_of_distance.append(points_of_radius[start_index])

    show_image(img)
    new_distance = [distance[0], distance[1]]
    for i in range(2, 360):
        if  new_distance[i-1] < 10 and abs(distance[i] - new_distance[i-1]) > 20:
            new_distance.append(new_distance[i-1])
            continue
        # Cách xa rìa cũ
        if abs(distance[i] - new_distance[i-1]) > 20:
            # Nếu ở gần tâm
            if distance[i]< 20:
                new_distance.append(new_distance[i-1])
                continue

            is_colony = False
            for j in range(7):
                id = i+j
                if id >= 360:
                    id -= 360
                if abs(distance[id] - new_distance[i-1]) < 30:
                    is_colony = True
                    break
            if is_colony :
                new_distance.append(new_distance[i-1])
                points_of_distance[i] = (x,y)
            else:
                new_distance.append(distance[i])
        else:
            new_distance.append(distance[i])
    # plot thành 2 biểu đồ để dễ nhìn
    

    plt.subplot(2, 1, 1)
    plt.plot(distance)
    plt.title('Original Distance')

    plt.subplot(2, 1, 2)
    plt.plot(new_distance)
    plt.title('Smoothed Distance')

    plt.tight_layout()
    plt.show()   
    img2 = cv2.imread(path)
    points_of_distance = []
    for angle in range(360):
        dist = radius - new_distance[angle]
        x = int(center[0] + dist * np.cos(angle * np.pi / 180))
        y = int(center[1] + dist * np.sin(angle * np.pi / 180))
        points_of_distance.append((x, y))
    
    distance = new_distance
    # distance = np.convolve(distance, [1/3, 1/3, 1/3], mode='same')
    smooth_distance = gaussian_filter(distance, sigma=6)
    shift_smooth_distance = np.roll(np.array(smooth_distance), 180)
    peaks, _ = find_peaks(-smooth_distance, prominence=10)
    shift_peaks, _ = find_peaks(-shift_smooth_distance,
                                prominence=10)
    shift_peaks = np.where(shift_peaks - 180 < 0,
                           shift_peaks + 180, shift_peaks - 180)
    peaks = np.unique(np.concatenate((peaks, shift_peaks)))

    valleys, _ = find_peaks(smooth_distance, prominence=10, height=50)
    shift_valleys, _ = find_peaks(
        shift_smooth_distance, prominence=10, height=50)
    shift_valleys = np.where(shift_valleys - 180 < 0,
                             shift_valleys + 180, shift_valleys - 180)
    valleys = np.unique(np.concatenate((valleys, shift_valleys)))

    plt.plot(smooth_distance)
    # plt.plot(shift_smooth_distance)
    plt.plot(peaks, smooth_distance[peaks], "x")
    plt.plot(valleys, smooth_distance[valleys], "o")
    plt.show()

    if len(peaks) != len(valleys):
        raise Exception("Số đỉnh và số thung lũng không bằng nhau")

    threshold_peak = []
    if peaks[0] > valleys[0]:
        for i in range(len(peaks)):
            threshold_left = peaks[i] - valleys[i]
            threshold_right = (
                valleys[i+1] if i < len(peaks) - 1 else valleys[0] + 360) - peaks[i]
            threshold_peak.append((peaks[i], threshold_left, threshold_right))
    else:
        for i in range(len(peaks)):
            threshold_left = peaks[i] - \
                (valleys[i-1] if i > 0 else valleys[-1] - 360)
            threshold_right = valleys[i] - peaks[i]
            threshold_peak.append((peaks[i], threshold_left, threshold_right))

    colonies = []

    for (peak, threshold_left, threshold_right) in threshold_peak:
        points_of_colony = []
        points_of_colony.append(points_of_distance[peak])

        threshold_index_left = np.mod(peak - threshold_left//2, 360)
        id_left = peak-1
        flag_out_of_index = False
        threshold_distance_bottom = smooth_distance[threshold_index_left]
        threshold_distance_top = smooth_distance[np.mod(
            threshold_index_left - threshold_left//2, 360)]
        list_point_lower = []
        list_point_upper = []
        while id_left > peak - threshold_left:
            if id_left < 0:
                id_left += 360
                flag_out_of_index = True
            if distance[id_left] < threshold_distance_bottom:
                list_point_lower.append(
                    (points_of_distance[id_left], distance[id_left]))
            elif distance[id_left] < threshold_distance_top:
                list_point_upper.append(
                    (points_of_distance[id_left], distance[id_left]))
            if flag_out_of_index:
                id_left -= 360
            id_left -= 1
        if len(list_point_upper) > 0:
            list_point_upper.sort(key=lambda x: x[1])
            points_of_colony.append(list_point_upper[0][0])
        elif len(list_point_lower) > 0:
            list_point_lower.sort(key=lambda x: -x[1])
            points_of_colony.append(list_point_lower[0][0])

        threshold_index_right = np.mod(peak + threshold_right//2, 360)
        id_right = peak+1
        flag_out_of_index = False
        threshold_distance_bottom = smooth_distance[threshold_index_right]
        threshold_distance_top = smooth_distance[np.mod(
            threshold_index_right + threshold_right//2, 360)]
        list_point_lower = []
        list_point_upper = []
        while id_right < peak + threshold_right:
            if id_right >= 360:
                id_right -= 360
                flag_out_of_index = True
            if distance[id_right] < threshold_distance_bottom:
                list_point_lower.append(
                    (points_of_distance[id_right], distance[id_right]))
            elif distance[id_right] < threshold_distance_top:
                list_point_upper.append(
                    (points_of_distance[id_right], distance[id_right]))
            if flag_out_of_index:
                id_right += 360
            id_right += 1

        if len(list_point_upper) > 0:
            list_point_upper.sort(key=lambda x: x[1])
            points_of_colony.append(list_point_upper[0][0])
        elif len(list_point_lower) > 0:
            list_point_lower.sort(key=lambda x: -x[1])
            points_of_colony.append(list_point_lower[0][0])

        colonies.append(points_of_colony)
    print("Point của nấm:", colonies)

    # lọc colonoies có phần tử có chiều dài lớn hơn 3
    colonies = [colony for colony in colonies if len(colony) >= 3]
    image = cv2.imread(path)
    for colony in colonies:
        for point in colony:
            image = draw_point(image, point)
    # show_image(image)

    circles = []
    for colony in colonies:
        center, radius = determine_colony(colony)
        image = cv2.circle(image, tuple(center), int(radius), (0, 255, 0), 2)
        circles.append((center, radius))
        # extract_colony(center, radius)/
    show_image(image)
    image = cv2.imread(path)
    masks = []
    for i, circle in enumerate(circles):
        mask = np.zeros(image.shape, dtype=np.uint8)
        center, radius = circle
        cv2.circle(mask, circle[0], circle[1], (255, 255, 255), -1, 8, 0)
        for j, other_circle in enumerate(circles):
            if i != j:
                point1, point2 = circle_intersection(circle, other_circle)
                if point1 is not None:
                    a, b = determine_line(point1, point2)
                    position_of_center = 1 if a * \
                        center[0] + b < center[1] else -1
                    sub_mask = np.zeros(image.shape, dtype=np.uint8)
                    for y in range(image.shape[0]):
                        for x in range(image.shape[1]):
                            position_of_pixel = 1 if a * x + b < y else -1
                            sub_mask[y, x] = (
                                255, 255, 255) if position_of_pixel == position_of_center else (0, 0, 0)
                    mask = cv2.bitwise_and(mask, sub_mask)
        masks.append(mask)
        masked_image = cv2.bitwise_and(image, mask)
        masked_image = masked_image[center[1]-radius:center[1] +
                                    radius, center[0]-radius:center[0]+radius]
        # write image
        image_name = path.split('/')[-1].split('_')[0] + '_' + str(i) + '.jpg'
        # cv2.imwrite('colonies/' + image_name, masked_image)
        show_image(masked_image)


path = get_path('new_data/YES/ob', 28)


def main():
    image = cv2.imread(path)
    # show_image(image)
    image = preprocess(image)

    # contour = determine_petri_dish(image)
    # image = cv2.imread(path)
    # draw_contour(image, contour)
    # show_image(image)
    histogram_distance(image)


if __name__ == '__main__':
    main()
