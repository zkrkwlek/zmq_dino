import threading
import time

class DownloadDataManager:
    def __init__(self, required_keywords):
        # 짝을 맞춰야 하는 필수 데이터 키워드 집합 (e.g., {b"image", b"salad_res"})
        self.required_keywords = set(required_keywords)
        # FID별 데이터 저장소
        # 구조: { fid: { "data": { b"image": data, b"salad_res": data }, "created_at": time.time() } }
        self.storage = {}
        self.lock = threading.Lock()

    def register_data(self, kw, src, fid, data):
        """데이터가 도착할 때마다 기록하고, 세트가 완성되면 전체 데이터를 반환합니다."""
        with self.lock:
            if fid not in self.storage:
                self.storage[fid] = {
                    "data": {},
                    "created_at": time.time()
                }

            # 도착한 데이터 저장
            self.storage[fid]["data"][kw] = (src, data)

            # 현재까지 모인 키워드 확인
            current_keywords = set(self.storage[fid]["data"].keys())

            # 모든 필수 데이터가 모였는지 검증
            if self.required_keywords.issubset(current_keywords):
                completed_set = self.storage[fid]["data"]
                del self.storage[fid]
                return completed_set  # { b"image": (src, img_bytes), b"salad_res": (src, salad_bytes) }

            return None

    def clear_old_data(self, expire_time_sec=15):
        """네트워크 문제로 짝이 안 맞춰진 채 방치된 데이터의 메모리(RAM) 누수를 방지합니다."""
        with self.lock:
            current_time = time.time()
            expired_fids = [
                fid for fid, info in self.storage.items()
                if (current_time - info["created_at"]) > expire_time_sec
            ]
            for fid in expired_fids:
                print(f"[-] [Data Timeout] {expire_time_sec}초 만료로 FID: {fid}의 미완성 데이터를 파기합니다.")
                del self.storage[fid]

class NotificationManager:
    def __init__(self, required_keywords):
        # 쌍을 맞춰야 하는 필수 키워드 집합 (e.g., {b"image", b"salad_res"})
        self.required_keywords = set(required_keywords)
        # FID별로 수신된 노티 정보를 저장할 딕셔너리
        # 구조: { fid: { b"image": (src, kw), b"salad_res": (src, kw) } }
        self.storage = {}
        self.lock = threading.Lock()  # 멀티스레드 안전용 (필요시)

    def register_notify(self, kw, src, fid):
        """노티가 올 때마다 기록하고, 모든 조건이 만족되면 다운로드할 리스트를 반환합니다."""
        with self.lock:
            if fid not in self.storage:
                self.storage[fid] = {
                    "data": {},
                    "created_at": time.time()
                }

            # 해당 FID에 키워드와 소스 저장
            self.storage[fid]["data"][kw] = (src, kw)

            # 현재 FID에 쌓인 키워드들이 필수 키워드를 모두 만족하는지 확인
            current_keywords = set(self.storage[fid]["data"].keys())

            if self.required_keywords.issubset(current_keywords):
                download_targets = list(self.storage[fid]["data"].values())
                del self.storage[fid]
                return download_targets

            return None

    def clear_old_fid(self, expire_time_sec=10):
        """네트워크 유실 등으로 인해 한쪽 노티가 영원히 안 와서 쌓이는 찌꺼기 메모리를 방지합니다."""
        with self.lock:
            current_time = time.time()
            # 딕셔너리 순회 중 삭제 시 에러가 발생하므로, 지울 대상을 리스트로 먼저 확보합니다.
            expired_fids = [
                fid for fid, info in self.storage.items()
                if (current_time - info["created_at"]) > expire_time_sec
            ]

            for fid in expired_fids:
                print(f"[-] [Timeout] 만료 시간({expire_time_sec}초) 초과로 FID: {fid}의 찌꺼기 노티 데이터를 해제합니다.")
                del self.storage[fid]