import numpy as np

# STL 모델 기준값 (P^t_s)
stl_values = {
    "semantic": 0.5082 * 100,  # mIoU는 퍼센트로 변환
    "depth": 0.4004,
    "normal": 23.8575
}

# log.txt 파일 읽기 (첫 번째 행 무시)
with open("/workspace/UniversalRepresentations/DensePred/results/abl/resnet50_deeplab_head_url_vanilla_uniform_alr_0.01_log.txt", "r") as f:
    lines = f.readlines()[1:]  # 첫 번째 행 제외

# 데이터 추출
data = []
for line in lines:
    values = line.strip().split()
    if len(values) >= 20:  # 최소한 20개 열이 있어야 함
        seg_miou = float(values[14]) * 100  # 14번째 열 (퍼센트 변환)
        depth_err = float(values[17])  # 17번째 열
        normal_err = float(values[20])  # 20번째 열
        data.append([seg_miou, depth_err, normal_err])

# NumPy 배열 변환
data = np.array(data)

# ΔMTL 계산
delta_mtl_values = (
    (data[:, 0] - stl_values["semantic"]) / stl_values["semantic"]
    - (data[:, 1] - stl_values["depth"]) / stl_values["depth"]
    - (data[:, 2] - stl_values["normal"]) / stl_values["normal"]
) / 3 * 100  # len(tasks) = 3

# 평균 ΔMTL이 가장 큰 행 찾기
max_delta_idx = np.argmax(delta_mtl_values)
max_delta_mtl = delta_mtl_values[max_delta_idx]  # 최대 ΔMTL 평균값
max_seg_miou, max_depth_err, max_normal_err = data[max_delta_idx]

# 결과 출력
print(f"가장 큰 ΔMTL 평균을 갖는 행:")
print(f"ΔMTL: {max_delta_mtl:.6f}")
print(f"Segmentation mIoU: {max_seg_miou:.6f}")
print(f"Depth Absolute Error: {max_depth_err:.6f}")
print(f"Normal mError: {max_normal_err:.6f}")
