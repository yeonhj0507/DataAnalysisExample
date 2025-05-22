# Data-Analysis-Examples

데이터 분석/머신러닝 입문자가 **핵심 기법**(PCA, K-Means, 선형·다중 회귀)을 빠르게 체험할 수 있도록 만든 예제 모음입니다.  
각 스크립트는 **표준 라이브러리 + scikit-learn**만으로 돌아갑니다.

## 포함된 기술
| 폴더 | 기술 | 설명 |
|------|------|------|
| `scripts/pca.py` | **PCA (주성분 분석)** | 고차원 데이터를 2D 로 투영 후 시각화 |
| `scripts/kmeans.py` | **K-Means 클러스터링** | 최적 군집 수 탐색 (`elbow`, `silhouette`) |
| `scripts/regression.py` | **선형·다중 회귀** | 캘리포니아 주택 가격 예측, 성능 리포트 |

## 빠른 시작

```bash
# 1) 가상환경 준비 (선택)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\Activate

# 2) 필수 패키지 설치
pip install -r requirements.txt

# 3) 예제 실행
python scripts/pca.py
python scripts/kmeans.py
python scripts/regression.py
