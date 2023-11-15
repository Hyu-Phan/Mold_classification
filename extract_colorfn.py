import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_center(image_path):
    # Đọc ảnh từ đường dẫn
    image=cv2.imread(image_path)
    height, width, _ = image.shape
    # xác định tâm, bán kính
    center=(int(width/2),int(height/2))
    radius=int(width/2*0.3)
    # Tạo mặt nạ hình tròn
    mask = np.zeros_like(image)
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
    # Áp dụng mặt nạ để cắt phần ảnh trong vòng tròn
    result = cv2.bitwise_and(image, mask)
    # average_color = np.mean(result, axis=(0, 1))
    # print("Đặc trưng màu trung bình:", average_color)
    # Hiển thị ảnh kết quả
    cv2.imshow('Cropped Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result

def extract_middle(image_path):
    # Đọc ảnh từ đường dẫn
    image=cv2.imread(image_path)
    height, width, _ = image.shape
    # Tạo mặt nạ hình tròn trong 
    mask_inner = np.zeros_like(image)
    center=(int(width/2),int(height/2))
    inner_radius=int(width/2*0.3)

    cv2.circle(mask_inner, center, inner_radius, (255, 255, 255), thickness=-1)

    # Tạo mặt nạ hình tròn ngoài (mặt nạ 2)
    outer_radius=int(width/2*0.8)
    mask_outer = np.zeros_like(image)
    cv2.circle(mask_outer, center, outer_radius, (255, 255, 255), thickness=-1)

    # Lấy khoảng viền ở mặt nạ ngoài không giao với mặt nạ trong
    outer_border = cv2.bitwise_xor(mask_outer, mask_inner)

    # Áp dụng mặt nạ để chọn phần ảnh trong khoảng viền
    outer_border_image = cv2.bitwise_and(image, outer_border)

    # average_color = np.mean(outer_border_image, axis=(0, 1))

    # print("Đặc trưng màu trung bình:", average_color)

    # Hiển thị ảnh và khoảng viền
  #  cv2.imshow('Original Image', image)
    cv2.imshow('Outer Border', outer_border_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return outer_border_image

def extract_border(image_path):
    # Đọc ảnh từ đường dẫn
    image=cv2.imread(image_path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    # Tạo mặt nạ hình tròn trong 
    mask_inner = np.zeros_like(image)
    center=(int(width/2),int(height/2))
    inner_radius=int(width/2*0.8)

    cv2.circle(mask_inner, center, inner_radius, (255, 255, 255), thickness=-1)

    # Tạo mặt nạ hình tròn ngoài (mặt nạ 2)
    outer_radius=int(width/2)
    mask_outer = np.zeros_like(image)
    cv2.circle(mask_outer, center, outer_radius, (255, 255, 255), thickness=-1)

    # Lấy khoảng viền ở mặt nạ ngoài không giao với mặt nạ trong
    outer_border = cv2.bitwise_xor(mask_outer, mask_inner)

    # Áp dụng mặt nạ để chọn phần ảnh trong khoảng viền
    outer_border_image = cv2.bitwise_and(image, outer_border)

    average_color = np.mean(outer_border_image, axis=(0, 1))
    avg1=np.mean(image, axis=(0, 1))
    # print("đặc trưng cả ảnh:",avg1)
    # print("Đặc trưng màu trung bình:", average_color)
    # Hiển thị ảnh và khoảng viền
#    cv2.imshow('Original Image', image)
    cv2.imshow('Outer Border', outer_border_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return outer_border_image

# Xác định màu sắc chủ đạo
# def dominant_color(image_path):
#     image = cv2.imread('C:\\Users\\Admin\\Desktop\\DATN\\Project\\DATN\\2.png')
#     pixels = image.reshape((-1, 3))
#     kmeans = KMeans(n_clusters=5)
#     kmeans.fit(pixels)
#     centers = kmeans.cluster_centers_
#     # Xác định màu sắc chủ đạo từ trung tâm của cụm có tần suất xuất hiện cao nhất
#     dominant_colors = [tuple(map(int, center)) for center in centers]
#     print("Dominant Colors:", dominant_colors)

def extract_color(image):
    color=[]
    avg1=np.mean(image, axis=(0, 1))
    image = cv2.imread(image_path)
    mean_color = np.mean(image, axis=(0, 1))  
    std_color = np.std(image, axis=(0, 1))    
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
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Tính histogram của ảnh xám
    hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    # Xác định giá trị có tần suất xuất hiện nhiều nhất (bin)
    dominant_gray_bin = np.argmax(hist_gray)
    color.append([mean_color,std_color,hue_mean,saturation_mean,value_mean,hue_std,saturation_std,value_std,dominant_gray_bin])
    return color

image_path = 'C:\\Users\\Admin\\Desktop\\DATN\\Project\\DATN\\2.png'
    # d={'avgR1':[],'avgG1':[],'avgB1':[],'avgR2':[],'avgG2':[],'avgB2':[],'avgR3':[],'avgG3':[],'avgB3':[],
    #    'avgH1':[],'avgS1':[],'avgV1':[],'avgH2':[],'avgS2':[],'avgV2':[],'avgH3':[],'avgS3':[],'avgV3':[],
    #    'stdR1':[],'stdG1':[],'stdB1':[],'stdR2':[],'stdG2':[],'stdB2':[],'stdR3':[],'stdG3':[],'stdB3':[],
    #    'stdH1':[],'stdS1':[],'stdV1':[],'stdH2':[],'stdS2':[],'stdV2':[],'stdH3':[],'stdS3':[],'stdV3':[],
    #    }
center=extract_center(image_path)
middle=extract_middle(image_path)
border=extract_border(image_path)
print(extract_color(center))
print(extract_color(middle))
print(extract_color(border))
