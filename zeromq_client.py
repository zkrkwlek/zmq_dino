import zmq
import zmq.asyncio
import asyncio
import cv2
import numpy as np
import os
import glob
import natsort
import argparse
import time

from geometry.reconstruction import Open3DReconstruction

#--resize --use_depth --use_gt

# 설정
SERVER_ADDR = "tcp://143.248.2.207:37001"
CLIENT_ID = b"SIM_PY_DEVICE_01"
SEND_KW = b"image"
RECV_KW = [] #[b"salad_res", b"xfeat_kp_res"]

#dataset_path = "F:\\SLAM_DATASET\\TUM\\rgbd_dataset_freiburg2_desk\\rgb"
#YAML_PATH = "F:\\SLAM_DATASET\\TUM\\TUM2.yaml"
dataset_path = "D:\\UVR\\simplerecon-main\\data_scripts\\ScanNetv2\\scans\\scene0256_00\\color"
YAML_PATH = "D:\\UVR\\simplerecon-main\\data_scripts\\ScanNetv2\\ScanNet.yaml"

class SimulationClient:
    def __init__(self, use_depth=False,use_resize=False,use_gt=False):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, CLIENT_ID)
        self.socket.connect(SERVER_ADDR)
        self.frame_count = 0

        # TUM 데이터셋 경로 설정 (예: 'TUM/rgbd_dataset_freiburg1_xyz/rgb')
        self.use_depth = use_depth
        self.use_gt = use_gt
        self.use_resize = use_resize
        self.dataset_path = dataset_path
        self.image_files = self.get_image_list()
        self.current_idx = 0

        if self.use_depth:
            self.depth_path = self.dataset_path.replace("color", "depth")
        if self.use_gt:
            self.pose_path = self.dataset_path.replace("color", "pose")

        # 카메라 파라미터 로드
        self.cam_params = self.load_camera_params(YAML_PATH)

    def load_camera_params(self, path):
        """YAML 파일에서 카메라 파라미터를 읽어옵니다."""
        #fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        import yaml
        with open(path, 'r') as stream:
            fs = yaml.full_load(stream)

        sx = 1.0
        sy = 1.0
        w = int(fs['Image.width'])
        h = int(fs['Image.height'])
        if self.use_resize:
            orig_w = float(fs['Image.width'])
            orig_h = float(fs['Image.height'])
            w = 640
            h = 480
            sx = 640.0 / orig_w
            sy = 480.0 / orig_h


        params = {
            "w": w,
            "h": h,
            "fx": fs['Camera.fx']*sx,
            "fy": fs['Camera.fy']*sy,
            "cx": fs['Camera.cx']*sx,
            "cy": fs['Camera.cy']*sy,
            "k1": fs['Camera.k1'],
            "k2": fs['Camera.k2'],
            "p1": fs['Camera.p1'],
            "p2": fs['Camera.p2'],
            "k3": 0.0
        }
        # k3가 있으면 읽고, 없으면 0.0으로 설정
        if 'Camera.k3' in fs.keys():
            params["k3"] = fs['Camera.k3']
        return params

    def get_image_list(self):
        """
        rgb.txt 파일을 읽어 이미지 경로 목록을 생성합니다.
        파일명의 .color.jpg 부분을 .color.640.jpg로 변경하여 읽습니다.
        """
        # rgb.txt는 보통 dataset_path(color 폴더)의 부모 폴더에 위치함
        base_dir = os.path.dirname(self.dataset_path)
        txt_path = os.path.join(base_dir, "rgb.txt")

        files = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                # 1. 처음 두 줄 무시 (헤더 처리)
                lines = f.readlines()[2:]

                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    # 2. 형식: "timestamp color/frame-000000.color.jpg"
                    parts = line.split()
                    if len(parts) >= 2:
                        rel_path = parts[1]  # "color/frame-000000.color.jpg"

                        # 3. .color.jpg -> .color.640.jpg 로 변경
                        # 만약 640 전용 파일이 따로 있다면 아래와 같이 치환합니다.
                        if self.use_resize:
                            rel_path = rel_path.replace(".color.jpg", ".color.640.png")

                        # 4. 절대 경로 생성 (base_dir + rel_path_640)
                        full_path = os.path.normpath(os.path.join(base_dir, rel_path))
                        files.append(full_path)

            print(f"[*] Successfully loaded {len(files)} images (640 version) from rgb.txt")
        else:
            print(f"[!] Error: Could not find rgb.txt at {txt_path}")
            # 파일이 없을 경우 예외 처리나 기존 glob 방식 활용 가능

        return files

    def get_image_list2(self):
        """폴더 내 이미지 목록을 가져와 정렬합니다."""
        # TUM 데이터셋은 .png 확장자를 주로 사용합니다.
        extensions = ['*.png', '*.jpg', '*.jpeg']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(self.dataset_path, ext)))

        # 타임스탬프 순서(숫자 순)로 정렬
        return natsort.natsorted(files)

    async def send_connect_info(self):
        """최초 1회 카메라 정보를 서버에 전송 (Action: NOTIFY, KW: connect)"""
        p = self.cam_params

        # [w, h, fx, fy, cx, cy, k1, k2, p1, p2, k3]
        cam_data = np.array([
            p["w"], p["h"],
            p["fx"], p["fy"], p["cx"], p["cy"],
            p["k1"], p["k2"], p["p1"], p["p2"], p["k3"]
        ], dtype=np.float32)  # C++ float와 대응

        # 바이너리 데이터로 변환하여 전송
        payload = cam_data.tobytes()

        # [Empty, Action, KW, Source, FID, Payload]
        await self.socket.send_multipart([
            b"", b"NOTIFY", b"connect", CLIENT_ID, b"0", payload
        ])
        print(f"[*] Sent camera intrinsics (connect) to server : {cam_data}")

    async def register(self):
        """서버에 수신 키워드 등록"""
        #for문으로 모든 키워드에 대응하도록 하기.
        #[ID, Empty, Action, KW, Source, Target]
        await self.socket.send_multipart([b"", b"RECV_REG"]+RECV_KW+[CLIENT_ID, b"MY_ONLY"])
        print(f"[*] {CLIENT_ID} registered to receive {RECV_KW}")

    async def upload_loop(self):
        """이미지 생성 및 3장당 1장 전송 루프"""
        # 1. 기기 정보(Connect) 먼저 전송
        await self.send_connect_info()
        await asyncio.sleep(0.1)  # 서버 처리 대기 시간

        # 2. 맵 초기화 알림
        self.socket.send_multipart([b"", b"NOTIFY", b"map_init", CLIENT_ID, b"0", b"none"])
        print(len(self.image_files), self.current_idx)
        while self.current_idx  < len(self.image_files):
            self.frame_count += 1

            # 가상의 이미지 생성 (또는 카메라 읽기)
            # 여기서는 검은색 배경에 프레임 번호가 적힌 이미지 생성
            #img = np.zeros((480, 640, 3), dtype=np.uint8)
            #cv2.putText(img, f"Frame: {self.frame_count}", (50, 200),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            rgb_path = self.image_files[self.current_idx]
            base_filename = os.path.basename(rgb_path)
            root_dir = os.path.dirname(self.dataset_path)

            img = cv2.imread(rgb_path)
            if img is None:
                continue
            self.current_idx += 1

            img_depth = None
            if self.use_depth:
                # 파일명 기반으로 depth 매칭 (예: 100.jpg -> 100.png)
                base_name = os.path.splitext(os.path.basename(rgb_path))[0]
                if self.use_resize:
                    base_name = base_name.replace("color", "depth").replace("depth.640", "depth")
                else:
                    base_name = base_name.replace("color", "depth")
                depth_file = os.path.join(self.depth_path, base_name + ".png")
                if os.path.exists(depth_file):
                    # ScanNet/TUM은 16bit PNG이므로 UNCHANGED로 읽어야 함
                    img_depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

            camera_pose = None
            if self.use_gt:
                if self.use_resize:
                    pose_name = base_filename.replace("color.640.png","pose.txt")
                else:
                    pose_name = base_filename.replace("color.png", "pose.txt")
                pose_path = os.path.join(self.pose_path, pose_name)
                if os.path.exists(pose_path):
                    try:
                        # ScanNet 포즈 파일은 4x4 행렬이 텍스트로 저장됨
                        pose_matrix = np.loadtxt(pose_path)
                        if pose_matrix.shape == (4, 4):
                            # Camera-to-World 행렬에서 R, t 추출
                            # R = pose_matrix[0:3, 0:3], t = pose_matrix[0:3, 3]
                            pose_matrix[0, 3] *= -1
                            pose_matrix[0:3, 0] *= -1
                            """
                            pose_matrix[1, 3] *= -1  # Y축 반전
                            pose_matrix[2, 3] *= -1  # Z축 반전

                            # 회전 변환 (Rotation) 수정
                            # Y, Z축 행과 열에 대해 부호를 교정해야 회전도 일치함
                            pose_matrix[0:3, 1] *= -1
                            pose_matrix[0:3, 2] *= -1
                            """
                            camera_pose = pose_matrix
                    except Exception as e:
                        print(f"[!] Pose load error: {e}")

            # 3장당 1장 전송
            if self.frame_count % 3 == 0:
                # 바이너리 직렬화 (JPEG)
                fid_bytes = str(self.frame_count).encode()
                _, img_encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                img_bytes = img_encoded.tobytes()

                # [Empty, Action, KW, Source, Target, ID, Payload]
                await self.socket.send_multipart([
                    b"", b"UPLOAD", SEND_KW, CLIENT_ID, fid_bytes, img_bytes
                ])
                if img_depth is not None:
                    # 16-bit 정보를 보존하기 위해 .png 무손실 인코딩 사용
                    _, depth_encoded = cv2.imencode('.png', img_depth)
                    await self.socket.send_multipart([
                        b"", b"UPLOAD", b"unidepth_res", CLIENT_ID, fid_bytes, depth_encoded.tobytes()
                    ])
                if camera_pose is not None:
                    # R | t (3x4 행렬) 추출
                    rt_matrix = camera_pose[:3, :4].astype(np.float32)
                    await self.socket.send_multipart([
                        b"", b"NOTIFY", b"gt_pose", CLIENT_ID, fid_bytes, rt_matrix.tobytes()
                    ])

            # 약 30 FPS 시뮬레이션
            await asyncio.sleep(0.033)

    async def receive_loop(self):
        """서버 알림 수신 및 데이터 다운로드 루프"""
        while True:
            msg = await self.socket.recv_multipart()
            # msg 구조: [Empty, Action, KW, Source, Target/ID, FrameID, (Payload)]
            #[ID, Empty, Action, KW, Source, Target, FID]
            #[ID, Empty, Action, KW, Source, Target, FID, Data]

            _, action, kw, src, fid = msg[:5]

            if action == b"NOTIFY":
                print(f"[!] Notification: {kw.decode()} from {src.decode()} for frame {fid.decode()}")

                # 알림을 받으면 즉시 DOWNLOAD 요청
                await self.socket.send_multipart([
                    b"", b"DOWNLOAD", kw,CLIENT_ID, fid
                ])

            elif action == b"DATA_REPLY":
                payload = msg[5]
                print(f"[v] Downloaded {kw.decode()} for frame {fid.decode()} (Size: {len(payload)} bytes)")
                # 여기서 받은 결과(Segmentation Mask 등)를 시각화하거나 처리합니다.

    async def run(self):
        await self.register()
        # 전송과 수신을 병렬로 실행
        await asyncio.gather(self.upload_loop(), self.receive_loop())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_depth', action='store_true', help='Enable depth image upload')
    parser.add_argument('--resize', action='store_true', help='Enable depth image upload')
    parser.add_argument('--use_gt', action='store_true', help='Enable depth image upload')
    args = parser.parse_args()

    recon = Open3DReconstruction()
    client = SimulationClient(use_depth=args.use_depth, use_resize = args.resize, use_gt = args.use_gt)

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("Simulation stopped.")

