from scannet.scannetexperiment import ScannetMultiViewExperimentEngine, ScannetExperimentVisualizer

if __name__ == "__main__":
    # 1. 오프라인 실험 인프라 경로 설정
    H5_PATH = "preprocessed_train_data.h5"
    SCANNET_ROOT = "./scans"

    # 2. 다중 뷰 실험 엔진 인스턴스화
    experiment_engine = ScannetMultiViewExperimentEngine(h5_path=H5_PATH, scans_root=SCANNET_ROOT)

    # 3. 현재 H5 저장소에서 선택 가능한 씬 아이디 리스트 추출 및 출력
    available_scenes = experiment_engine.get_available_scenes()
    print("=" * 60)
    print("🌲 [ScanNet 오프라인 실험실] 선택 가능한 전처리 완료 씬 ID 리스트:")
    for idx, s_id in enumerate(available_scenes):
        print(f"  [{idx}] {s_id}")
    print("=" * 60)

    # 4. 💡 [박사님 제어부]: 리스트 중 실험 및 검증을 원하는 특정 씬 아이디 지정
    # 예시: 첫 번째 씬을 타겟으로 잡거나 문자열로 직접 지정 가능
    SELECTED_SCENE = available_scenes[4]  # 또는 "scene0000_00" 직접 기입 가능

    # 5. 지정된 씬을 기반으로 내부 프레임 순회 및 타 뷰 이미지 비교 결합 파이프라인 구동
    experiment_engine.run_scene_multi_view_experiment(
        scene_id=SELECTED_SCENE,

        alpha=0.20,  # 소외 영역 보존을 위한 어텐션 바닥 값
        tau=0.75  # 최종 전역 그래프 이진화를 위한 매칭 임계값
    )