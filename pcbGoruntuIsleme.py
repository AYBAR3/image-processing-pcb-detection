import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
from shapely.geometry import box

# Veri yolları
reference_image_path = r"C:\Users\NFS\Desktop\PCB_DATASET\Reference\01.JPG"
test_image_paths = {
    "Missing Hole": r"C:\Users\NFS\Desktop\PCB_DATASET\rotation\Missing_hole_rotation",
    "Mouse Bite": r"C:\Users\NFS\Desktop\PCB_DATASET\rotation\Mouse_bite_rotation",
    "Open Circuit": r"C:\Users\NFS\Desktop\PCB_DATASET\rotation\Open_circuit_rotation"
}
annotation_paths = {
    "Missing Hole": r"C:\Users\NFS\Desktop\PCB_DATASET\Annotations\Missing_hole",
    "Mouse Bite": r"C:\Users\NFS\Desktop\PCB_DATASET\Annotations\Mouse_bite",
    "Open Circuit": r"C:\Users\NFS\Desktop\PCB_DATASET\Annotations\Open_circuit"
}

# Fonksiyon: Görüntüleri yükleme
def load_images(image_dir):
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(image_dir, filename)
            images.append((filename, cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)))
    return images

# Fonksiyon: XML Anotasyonlarını yükleme
def load_annotations(annotation_dir):
    annotations = {}
    for filename in os.listdir(annotation_dir):
        if filename.endswith(".xml"):
            xml_path = os.path.join(annotation_dir, filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            annotation_data = []
            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                if bbox is not None:
                    x_min = int(bbox.find("xmin").text)
                    y_min = int(bbox.find("ymin").text)
                    x_max = int(bbox.find("xmax").text)
                    y_max = int(bbox.find("ymax").text)
                    annotation_data.append((x_min, y_min, x_max, y_max))
            annotations[filename] = annotation_data
    return annotations

# Fonksiyon: Görüntü Çakıştırma (Image Registration)
def align_images(ref_img, test_img):
    print("Görüntüler hizalanıyor...")
    try:
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-5)

        # Histogram eşitleme ve bulanıklaştırma
        ref_img = cv2.equalizeHist(ref_img)
        test_img = cv2.equalizeHist(test_img)
        test_img = cv2.GaussianBlur(test_img, (5, 5), 0)

        print("Hizalama matrisi hesaplanıyor...")
        cc, warp_matrix = cv2.findTransformECC(ref_img, test_img, warp_matrix, warp_mode, criteria)
        print(f"Hizalama başarılı. Correlation Coefficient: {cc:.4f}")

        aligned_img = cv2.warpAffine(test_img, warp_matrix, (ref_img.shape[1], ref_img.shape[0]), flags=cv2.INTER_LINEAR)
        return aligned_img

    except cv2.error as e:
        print(f"Hizalama başarısız: {e}. Orijinal test görüntüsü ile devam ediliyor.")
        return test_img

# Fonksiyon: IoU Hesaplama
def calculate_iou(boxA, boxB):
    rect1 = box(boxA[0], boxA[1], boxA[2], boxA[3])
    rect2 = box(boxB[0], boxB[1], boxB[2], boxB[3])
    intersection = rect1.intersection(rect2).area
    union = rect1.union(rect2).area
    return intersection / union

# Fonksiyon: Doğruluk Analizi
def evaluate_defects(detected_boxes, annotated_boxes, iou_threshold=0.5):
    TP = 0  # True Positive
    FP = 0  # False Positive
    FN = 0  # False Negative

    matched_annotations = set()
    for detected in detected_boxes:
        found_match = False
        for i, annotated in enumerate(annotated_boxes):
            if i not in matched_annotations:
                iou = calculate_iou(detected, annotated)
                if iou >= iou_threshold:
                    TP += 1
                    matched_annotations.add(i)
                    found_match = True
                    break
        if not found_match:
            FP += 1

    FN = len(annotated_boxes) - len(matched_annotations)

    # Performans Metrikleri
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    accuracy = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0

    return {"TP": TP, "FP": FP, "FN": FN, "Precision": precision, "Recall": recall, "Accuracy": accuracy}

# Fonksiyon: Kusurları Görselleştirme ve Değerlendirme
def process_image(reference_image, test_image, annotated_boxes, defect_type, filename):
    print(f"Referans ve test görüntüsü hizalanıyor: {filename}")
    aligned_test_image = align_images(reference_image, test_image)
    print(f"Görüntüler hizalandı: {filename}")

    print("Farklar hesaplanıyor...")
    difference = cv2.absdiff(reference_image, aligned_test_image)

    print("Fark görüntüsü eşikleniyor...")
    _, binary_diff = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

    print("Konturlar tespit ediliyor...")
    contours, _ = cv2.findContours(binary_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Tespit edilen kontur sayısı: {len(contours)}")

    detected_boxes = []
    output_image = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Minimum alan filtresi
            x, y, w, h = cv2.boundingRect(contour)
            detected_boxes.append((x, y, x + w, y + h))
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Mavi kutular

    # Anotasyonları görselleştirme
    for (x_min, y_min, x_max, y_max) in annotated_boxes:
        cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Yeşil kutular

    # Fark ve eşik görüntülerini kaydetme
    cv2.imwrite(f"difference_{filename}.png", difference)
    cv2.imwrite(f"binary_diff_{filename}.png", binary_diff)

    print(f"Tespit edilen kutular: {detected_boxes}")

    return output_image, detected_boxes

# 1. Referans Görüntüsünü Yükleme
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

# 2. Test Görüntülerini ve Anotasyonları Yükleme
for defect_type, test_path in test_image_paths.items():
    test_images = load_images(test_path)
    annotations = load_annotations(annotation_paths[defect_type])

    print(f"\n{defect_type} için görüntüler ve anotasyonlar işleniyor...")
    print(f"Yüklenen test görüntüleri: {len(test_images)}")
    print(f"Yüklenen anotasyonlar: {len(annotations)}")

    for filename, test_image in test_images:
        xml_filename = filename.replace(".jpg", ".xml").replace(".png", ".xml")
        
        if xml_filename in annotations:
            print(f"\n{filename} işleniyor...")
            
            try:
                output_image, results = process_image(reference_image, test_image, annotations[xml_filename], defect_type, filename)
                print(f"{filename} sonuçları: {results}")

                # Görselleştirme
                cv2.imshow(f"{defect_type} - {filename}", cv2.resize(output_image, None, fx=0.5, fy=0.5))
                cv2.waitKey(0)
            except Exception as e:
                print(f"{filename} işlenemedi: {e}. İşlem atlanıyor.")
        else:
            print(f"Anotasyon bulunamadı: {filename}")
            
cv2.destroyAllWindows()
