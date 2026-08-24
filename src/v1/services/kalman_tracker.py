"""
kalman_tracker.py  (vendored from tracker_py/src/v1/services/tracker.py)
=======================================================================

추론 서비스(tracker_py)의 커스텀 칼만 트래커를 **그대로 이식**한 것이다. 자동 라벨의
트래킹(track_id 부여)을 운영 추론과 똑같은 규칙으로 하려는 목적이며, NIT_train 이
추론 서비스와 분리 배포될 때 tracker_py 를 런타임 의존하지 않도록 파일을 복사해 둔다.
원본을 고치면 이 파일도 다시 복사해 동기화한다(numpy + scipy 만 필요).

칼만 필터(Kalman Filter) 기반 다중 객체 추적기.

핵심 아이디어
-----------
1. [예측] 매 프레임마다 칼만 필터로 다음 위치를 예측 (등속 운동 모델).
2. [연결] 헝가리안 알고리즘으로 "예측 위치 ↔ 탐지 결과" 를 최적 매칭.
   - IoU 우선 → IoU 부족 시 중심점 거리(centroid distance)로 보완.
3. [스무딩] 레이블은 최근 N 프레임의 다수결(majority vote)로 안정화
   (가끔 다른 탱크 종류로 오인식되어도 안정적 레이블 유지).
4. [고스트 트래킹] 탐지 미검출 시에도 max_age 프레임 동안 칼만 예측값으로 track 유지.

출력 track dict
--------------
{
    "track_id"   : int,          # 영상 전체에서 고유 ID
    "label"      : str,          # 다수결 스무딩 레이블
    "score"      : float,        # 최근 탐지 신뢰도 평균
    "bbox"       : [x1,y1,x2,y2], # 칼만 스무딩 좌표
    "is_predicted": bool,        # True = 이번 프레임 탐지 없이 예측으로만 유지
    "status"     : str,          # 표시 상태: tentative/stable/locked/ghost/revived
    "age"        : int,          # 트랙 생성 후 경과 프레임
    "hit_streak" : int,          # 연속 탐지 프레임 수
    "vx"         : float,        # x 방향 속도 (px/frame)
    "vy"         : float,        # y 방향 속도 (px/frame)
}
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque, Counter


# ── 트래커 설정 (카메라/시나리오별로 조정) ──────────────────────────────────
@dataclass
class TrackerConfig:
    """
    트래커 가중치 묶음. 카메라(해상도/FPS/시점)에 따라 조정.

    사용 예:
        cfg = TrackerConfig(max_age=120, vel_alpha=0.30, arrow_scale=5.0)
        tracker = MultiObjectTracker(config=cfg)

    필드 의미는 `jo/docs/tracker_tuning.md` 참고.
    """

    # ── 칼만 측정 노이즈 R (해상도에 비례 — 가로 폭 / 640 비율로 스케일) ──
    pos_meas_noise:  float = 2.0   # cx, cy 측정 노이즈
    size_meas_noise: float = 8.0   # w,  h  측정 노이즈

    # ── 칼만 프로세스 노이즈 Q ──────────────────────────────────────────
    pos_proc_noise:   float = 6.0   # 위치 변화 허용 (↑ → 급변/방향전환에 빨리 적응 + 게이트 넓어짐)
    size_proc_noise:  float = 1.0   # 박스 크기 변화 허용
    vel_proc_noise:   float = 0.002 # 속도 변화 허용 (↓ → 화살표 안정/느림, ↑ → 반응 빠름) — 기존 0.01의 1/5
    vsize_proc_noise: float = 0.001 # 크기 변화율

    # ── 예측 시 속도 반영 비율 (방향 전환 대응) ─────────────────────────
    # 예측 위치 = 마지막 위치 + γ × 속도.
    # 1.0=등속 외삽(반전 시 예측이 옛 방향으로 날아가 ID 끊김),
    # <1.0=마지막 위치 근처에 머물러 방향 전환 후에도 같은 ID 매칭이 쉬워짐.
    # adaptive_damping=False 면 이 값을 고정 감쇠로 사용(기존 동작).
    vel_damping: float = 0.45

    # ── 적응형 감쇠 γ_t (포인트 D, 특허 독립항1) ────────────────────────
    # 트랙 상태에 따라 매 프레임 감쇠를 조절한다.
    #   · 연속 매칭·저불확실성 → γ_t ≈ vel_damping_max(=기존 0.45) → 기존 동작 보존
    #   · 미검출(가림) 길수록 / 위치 불확실성(P) 클수록 → γ_t ↓ (과외삽 억제, ID 유지)
    # γ_t = clip( vel_damping · (1 - damp_miss_decay·miss) · min(1, damp_p_ref/P_pos),
    #             vel_damping_min, vel_damping_max )
    adaptive_damping: bool  = True   # False → vel_damping 고정(레거시)
    vel_damping_min:  float = 0.10   # γ_t 하한 (과외삽 방지 최소 추종)
    vel_damping_max:  float = 0.45   # γ_t 상한 (연속 매칭 시 값)
    damp_miss_decay:  float = 0.15   # 미검출 1프레임당 감쇠 감소율(0~1)
    damp_p_ref:       float = 200.0  # 위치 불확실성 기준(P_cx+P_cy). 초과분만큼 γ_t↓ (0=비활성)

    # ── Detection 사전 필터 (트래커 진입 전에 가짜 박스 차단) ────────────
    det_score_min: float = 0.0  # YOLO conf 가 이 미만이면 무시 (0 = 비활성)
    min_bbox_area: float = 0.0  # bbox 픽셀 면적이 이 미만이면 무시 (0 = 비활성)

    # ── 매칭 비용행렬 ──────────────────────────────────────────────────
    base_gate:               float = 24.0  # 마할라노비스 게이트 (커질수록 매칭 너그러움)
    confirmed_gate_mult:     float = 1.3   # 확정 트랙(hit_streak ≥ min_hits) 게이트 배율
    label_match_bonus:       float = 0.15  # 같은 라벨일 때 비용 감소
    label_mismatch_penalty:  float = 0.20  # 다른 라벨일 때 비용 증가 (완전차단 X)
    iou_bonus_factor:        float = 0.3   # IoU 1.0 일 때 최대 감소량
    # IoU 2차 매칭 게이트: 마할라노비스 게이트를 벗어나 미매칭된 (고스트)트랙도
    # detection 과 박스가 이 IoU 이상 겹치면 그 detection 으로 갱신(ID 유지·중복 제거).
    iou_merge_gate:          float = 0.75  # 0 이면 비활성

    # ── 트랙 생명주기 (FPS 에 비례 — 30fps 기준) ────────────────────────
    max_age:  int = 100  # 미탐지 N 프레임까지 고스트로 유지 (≈3.3초@30fps)
    min_hits: int = 2    # N 프레임 연속 탐지 후 표시 시작

    # ── Re-ID (죽은 트랙 재식별: 같은 객체 재등장 시 ID 부활) ────────────
    # max_age 를 넘겨 죽은 트랙도 reid_max_age 동안은 "기억" 했다가,
    # 라벨이 같고 위치/크기가 비슷한 detection 이 나타나면 같은 ID 로 되살린다.
    # (모션 기반 — 마지막 위치 + 속도 외삽 거리 + 크기 비율 + 라벨)
    reid_enabled:         bool  = True
    reid_max_age:         int   = 300    # 죽은 트랙 기억 프레임 수 (≈5초@60fps)
    reid_dist_gate:       float = 150.0  # 부활 허용 기본 거리(px)
    reid_drift_per_frame: float = 5.0    # 경과 프레임당 허용거리 증가(px) — 오래될수록 너그럽게
    reid_size_ratio:      float = 0.4    # 박스 면적 비율 하한(0~1) — 너무 다른 크기는 거절
    # Re-ID 예측 위치 산출 시 '속도 외삽'을 신뢰하는 상한 프레임 수(G_cap).
    # 클수록 빠른 직진 표적을 멀리 예측(고속·근거리 유리), 작을수록 곡선/정지
    # 표적의 오예측 억제(원거리·기동 유리). 권장 범위 5~30 (고FPS일수록 크게).
    reid_vel_cap:         int   = 15
    # 라벨 일치 필수 여부. False 면 YOLO 가 라벨을 잘못 달아도(오탐) 위치/크기로 같은
    # 객체라 판단되면 이전 ID 를 부활시킨다(라벨 다르면 mismatch_pen 만큼 덜 선호).
    reid_require_label:        bool  = False
    reid_label_mismatch_pen:   float = 0.2   # 라벨 불일치 시 부활 비용 가산(0~1, 클수록 라벨 중시)

    # ── 스무딩 ────────────────────────────────────────────────────────
    vel_alpha:          float = 0.02  # 속도 EMA 계수 (낮을수록 부드러움/방향 안정)
    label_history_size: int   = 15    # 라벨 다수결 윈도우
    # 라벨 고정: 표가 label_lock_min_count 만큼 모이면 그 시점 다수결 라벨로 '고정'.
    # 이후 YOLO 가 라벨을 바꿔 추정해도(오탐) 트랙 라벨/이름은 변하지 않는다.
    label_lock:           bool = True
    label_lock_min_count: int  = 6    # 고정에 필요한 누적 라벨 샘플 수
    # 예측에 쓰는 속도를 '최근 N 프레임 평균 이동속도'로 스무딩(순간 가속/점프 억제).
    vel_smooth_window:  int   = 6     # 평균 이동속도 계산 윈도우(프레임). 2 이상.
    # 예측 속도 상한(px/frame). 한 번의 오탐으로 트랙이 '퓽' 날아가는 것을 방지.
    max_speed:          float = 0.5
    # ── 이동 데드존 (정지/미세이동 표적 지터 억제) ─────────────────────
    # 회귀 윈도우 내 총 변위가 (motion_dead_zone_ratio × 박스크기) 미만이면
    # 속도를 0 으로 간주 → 정지 표적이 지터로 '떠는' 것을 막는다.
    # 0 = 비활성(기존 동작). 고정형 CCTV 등 정지 표적이 많을 때 0.02~0.1 권장.
    motion_dead_zone_ratio: float = 0.0

    # ── 시각화 (해상도에 비례) ─────────────────────────────────────────
    arrow_scale:  float = 0.2  # 화살표 길이 배율 (px = 속도 × 이 값) — 추가 하향(0.5→0.2)
    speed_thresh: float = 0.3  # 이 미만이면 정지로 간주, 화살표 생략

    # ── 이동 경로(trail) ──────────────────────────────────────────────
    # 최근 N 프레임 동안의 박스 중심점을 저장 → 폴리라인으로 표시.
    # 0 이면 비활성. FPS 에 비례 조정 (예: 30fps → 30이면 약 1초)
    trail_length:    int = 20
    trail_thickness: int = 1   # 폴리라인 두께

    # ── 레거시 호환 (현재 비용행렬에 미사용, 추후 백오프 매칭에 활용) ──
    iou_threshold: float = 0.15
    max_dist_px:   float = 200.0

    def copy(self, **overrides) -> "TrackerConfig":
        """일부 값만 바꾼 새 config 반환."""
        return replace(self, **overrides)

    def as_dict(self) -> dict:
        return asdict(self)


# ── 좌표 변환 헬퍼 ────────────────────────────────────────────────────────────

def _to_cxcywh(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = float(x2 - x1)
    h  = float(y2 - y1)
    return cx, cy, w, h


def _to_xyxy(cx, cy, w, h):
    w = max(w, 1.0)
    h = max(h, 1.0)
    return [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)]


def _iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def _bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    w = max(0.0, float(x2 - x1))
    h = max(0.0, float(y2 - y1))
    return w * h


def _centroid_dist(b1, b2):
    cx1 = (b1[0] + b1[2]) / 2.0; cy1 = (b1[1] + b1[3]) / 2.0
    cx2 = (b2[0] + b2[2]) / 2.0; cy2 = (b2[1] + b2[3]) / 2.0
    return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))


# ── 단일 객체 칼만 트래커 ─────────────────────────────────────────────────────

class KalmanBoxTracker:
    """
    상태 벡터: [cx, cy, w, h, vx, vy, vw, vh]  (8차원)
    측정 벡터: [cx, cy, w, h]                  (4차원)
    등속 운동 모델 (constant velocity).
    """
    _next_id = 0

    def __init__(
        self,
        bbox,
        label: str,
        score: float,
        config: Optional[TrackerConfig] = None,
    ):
        self.config = config or TrackerConfig()

        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1

        self._init_kalman(bbox)

        self.time_since_update = 0   # 마지막 매칭 이후 경과 프레임
        self.hit_streak = 0          # 연속 탐지 횟수
        self.age = 0                 # 총 생존 프레임

        # 레이블 스무딩: config.label_history_size 프레임 다수결
        hist_size = max(1, int(self.config.label_history_size))
        self._label_hist = deque(maxlen=hist_size)
        self._label_hist.append(label)
        self._score_hist = deque(maxlen=hist_size)
        self._score_hist.append(score)
        # 라벨 고정값(None=아직 미고정). 충분한 표가 모이면 확정되어 더 안 바뀐다.
        self._locked_label: Optional[str] = None

        # 속도 EMA(지수이동평균) — 화면에 보여줄 속도 안정화용
        self._vx_ema = 0.0
        self._vy_ema = 0.0

        # 평균 이동속도 스무딩용 (age, cx, cy) 이력 — 실탐지 갱신 시에만 누적.
        # 속도 = (위치 변화) / (경과 '프레임' 수). age 는 매 프레임 predict 에서 +1 되므로
        # 중간 미탐지(빈 프레임)까지 포함한 실제 경과 프레임으로 나눠 속도를 정확히 구한다.
        vw = max(2, int(self.config.vel_smooth_window))
        self._cxcy_hist: deque = deque(maxlen=vw)
        _cx0, _cy0, _, _ = _to_cxcywh(bbox)
        self._cxcy_hist.append((0, float(_cx0), float(_cy0)))

        # 이동 경로(trail) — 최근 N 프레임 박스 중심점.
        # _collect_tracks 단계에서 매 프레임 한 번씩 추가됨.
        trail_n = max(1, int(self.config.trail_length))
        self._pos_history: deque = deque(maxlen=trail_n)

        # Re-ID 대기(죽은 트랙) 경과 프레임. _lost 버퍼에서만 증가.
        self._lost_gap = 0

        # 적응형 감쇠 γ_t 최근값(디버그/표시용). predict 마다 갱신.
        self._gamma_last = float(getattr(self.config, "vel_damping", 0.45))
        # 재식별(부활) 직후 status='revived' 로 표시할 잔여 프레임 수.
        self._revived_ttl = 0

    # ── 칼만 필터 초기화 ──────────────────────────────────────────────────────

    def _init_kalman(self, bbox):
        cx, cy, w, h = _to_cxcywh(bbox)
        n, m = 8, 4

        # 상태 전이 행렬 F: position += vel_damping × velocity
        # (vel_damping<1 → 예측이 옛 방향으로 덜 날아가 방향 전환에도 ID 유지가 쉬움)
        damp = float(getattr(self.config, "vel_damping", 1.0))
        self.F = np.eye(n)
        for i in range(m):
            self.F[i, i + m] = damp

        # 측정 행렬 H
        self.H = np.zeros((m, n))
        for i in range(m):
            self.H[i, i] = 1.0

        c = self.config

        # 측정 노이즈 R (픽셀 단위 불확실성) — 해상도에 비례 조정
        self.R = np.diag([
            c.pos_meas_noise,  c.pos_meas_noise,
            c.size_meas_noise, c.size_meas_noise,
        ])

        # 프로세스 노이즈 Q — 위치/크기/속도/크기변화율 분리
        self.Q = np.diag([
            c.pos_proc_noise,   c.pos_proc_noise,
            c.size_proc_noise,  c.size_proc_noise,
            c.vel_proc_noise,   c.vel_proc_noise,
            c.vsize_proc_noise, c.vsize_proc_noise,
        ])

        # 초기 공분산 P (속도 불확실성 크게)
        self.P = np.diag([10.0, 10.0, 20.0, 20.0, 1e4, 1e4, 1e3, 1e3])

        # 상태 초기화
        self.x = np.zeros((n, 1))
        self.x[:4, 0] = [cx, cy, w, h]

    def apply_config(self) -> None:
        """런타임 config 변경을 칼만 행렬(F/R/Q)에 반영한다(상태 x·P 는 보존).

        base_gate·라벨 보너스·vel_alpha 등은 매 프레임 config 에서 직접 읽으므로
        즉시 반영되지만, F/R/Q 는 init 때 1회 구성되므로 여기서 갱신해 준다.
        """
        c = self.config
        damp = float(getattr(c, "vel_damping", 1.0))
        for i in range(4):
            self.F[i, i + 4] = damp
        self.R = np.diag([
            c.pos_meas_noise,  c.pos_meas_noise,
            c.size_meas_noise, c.size_meas_noise,
        ])
        self.Q = np.diag([
            c.pos_proc_noise,   c.pos_proc_noise,
            c.size_proc_noise,  c.size_proc_noise,
            c.vel_proc_noise,   c.vel_proc_noise,
            c.vsize_proc_noise, c.vsize_proc_noise,
        ])

    # ── 적응형 감쇠 γ_t (포인트 D) ─────────────────────────────────────────────

    def _gamma_t(self) -> float:
        """트랙 상태에 따라 예측 감쇠 γ_t 를 계산한다.

        연속 매칭·저불확실성 트랙은 γ_t ≈ vel_damping_max(=기존 0.45)로 기존
        동작을 그대로 보존하고, 가림(미검출)이 길거나 위치 불확실성(P)이 커질수록
        γ_t 를 낮춰 옛 방향으로의 과도한 외삽을 억제한다(방향전환·가림에서 ID 유지).
        adaptive_damping=False 면 고정 감쇠(vel_damping)로 폴백한다.
        """
        c = self.config
        base = float(getattr(c, "vel_damping", 0.45))
        if not bool(getattr(c, "adaptive_damping", False)):
            return base
        g = base
        # (1) 미검출(가림)이 길수록 감쇠↓ — 옛 방향 과외삽 억제
        miss = int(self.time_since_update)
        if miss > 0:
            g *= max(0.0, 1.0 - float(getattr(c, "damp_miss_decay", 0.15)) * miss)
        # (2) 위치 불확실성(P_cx+P_cy)이 기준보다 크면 초과분만큼 감쇠↓
        p_ref = float(getattr(c, "damp_p_ref", 0.0))
        if p_ref > 0.0:
            p_pos = float(self.P[0, 0] + self.P[1, 1])
            if p_pos > p_ref:
                g *= p_ref / p_pos
        gmin = float(getattr(c, "vel_damping_min", 0.10))
        gmax = float(getattr(c, "vel_damping_max", base))
        return float(min(gmax, max(gmin, g)))

    # ── 예측 단계 ─────────────────────────────────────────────────────────────

    def predict(self):
        """칼만 예측 후 bbox 반환. 호출마다 age 1 증가."""
        # 크기가 0 이하로 내려가지 않도록 보호
        if self.x[2, 0] <= 0:
            self.x[2, 0] = 1.0
        if self.x[3, 0] <= 0:
            self.x[3, 0] = 1.0

        # 적응형 감쇠 γ_t 를 상태전이 F 의 속도항에 반영(매 프레임).
        gamma = self._gamma_t()
        self._gamma_last = gamma
        for i in range(4):
            self.F[i, i + 4] = gamma

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        if self._revived_ttl > 0:
            self._revived_ttl -= 1
        return self.bbox

    # ── 갱신 단계 ─────────────────────────────────────────────────────────────

    def update(self, bbox, label: str, score: float):
        """새 탐지 결과로 칼만 필터를 갱신."""
        z = np.array(_to_cxcywh(bbox), dtype=float).reshape(4, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hit_streak += 1
        self._label_hist.append(label)
        self._score_hist.append(score)

        # ── 라벨 고정: 충분한 표가 모이면 다수결 라벨로 확정(이후 불변) ──────────
        if (self.config.label_lock and self._locked_label is None
                and len(self._label_hist) >= max(1, int(self.config.label_lock_min_count))):
            self._locked_label = Counter(self._label_hist).most_common(1)[0][0]

        # ── 최소자승 회귀 기울기로 속도 추정 (detection jitter 억제) ──────────
        # 양 끝점 차분은 단 2개 샘플의 노이즈에 그대로 노출돼, 정지 객체도 끝점이
        # 우연히 어긋나면 가짜 속도가 잡힌다. 대신 윈도우 내 모든 (age, cx, cy) 점에
        # 직선을 최소자승 피팅한 기울기(px/frame)를 속도로 쓴다. N 점 전부를 쓰므로
        # 평균이 0인 측정 노이즈가 상쇄되고, age(=빈 프레임 포함 실제 경과)를 시간축에
        # 넣으므로 가려졌다 다시 잡혀도 속도가 올바르게 가중된다.
        cx_now = float(self.x[0, 0]); cy_now = float(self.x[1, 0])
        self._cxcy_hist.append((self.age, cx_now, cy_now))
        n = len(self._cxcy_hist)
        if n >= 2:
            ts = np.fromiter((p[0] for p in self._cxcy_hist), dtype=float, count=n)
            xs = np.fromiter((p[1] for p in self._cxcy_hist), dtype=float, count=n)
            ys = np.fromiter((p[2] for p in self._cxcy_hist), dtype=float, count=n)
            dt = ts - ts.mean()
            denom = float(dt @ dt)  # Σ(tᵢ−t̄)²
            if denom > 1e-9:
                avg_vx = float(dt @ (xs - xs.mean()) / denom)
                avg_vy = float(dt @ (ys - ys.mean()) / denom)
            else:
                avg_vx = avg_vy = 0.0
        else:
            avg_vx = avg_vy = 0.0

        # ── 이동 데드존: 윈도우 총 변위가 (κ × 박스크기) 미만이면 정지로 간주 ──
        # 정지/미세이동 표적이 detection jitter 로 '떠는' 것을 막는다.
        # κ=motion_dead_zone_ratio (0=비활성). 고정형 CCTV 등에서 유용.
        dz = float(getattr(self.config, "motion_dead_zone_ratio", 0.0))
        if dz > 0.0 and n >= 2:
            span = float(((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2) ** 0.5)
            size = 0.5 * (float(self.x[2, 0]) + float(self.x[3, 0]))
            if size > 0.0 and span < dz * size:
                avg_vx = 0.0
                avg_vy = 0.0

        vmax = float(self.config.max_speed)
        if vmax > 0:
            avg_vx = max(-vmax, min(vmax, avg_vx))
            avg_vy = max(-vmax, min(vmax, avg_vy))

        # 예측 속도 상태를 평균 이동속도로 대체 → 다음 predict 가 안정적으로 외삽.
        self.x[4, 0] = avg_vx
        self.x[5, 0] = avg_vy

        # 출력(화살표)용 EMA 도 평균 이동속도 기준으로 부드럽게.
        a = float(self.config.vel_alpha)
        self._vx_ema = (1.0 - a) * self._vx_ema + a * avg_vx
        self._vy_ema = (1.0 - a) * self._vy_ema + a * avg_vy

    # ── 재식별(Re-ID) 부활 ──────────────────────────────────────────────────────

    def revive(self, bbox, label: str, score: float):
        """죽었던 트랙을 새 detection 위치로 되살린다 (ID·라벨 이력 유지).

        오랜 공백 후라 속도는 신뢰할 수 없으므로 칼만을 새 위치로 재초기화하되,
        track_id 와 라벨 다수결 이력은 보존해 "같은 객체"로 이어지게 한다.

        부활 시점의 detection 라벨은 YOLO 오탐일 수 있으므로 **기존 라벨 이력을
        덮어쓰지 않는다**(같은 객체면 이전 라벨/ID 를 그대로 유지). 단, 부활 detection
        라벨이 기존 다수결 라벨과 같으면 그 신뢰를 보강하기 위해 한 표 추가한다.
        """
        self._init_kalman(bbox)
        self.time_since_update = 0
        self.hit_streak = max(1, int(self.config.min_hits))
        # 라벨이 일치할 때만 이력에 반영(오탐 라벨로 정체성이 바뀌는 것 방지).
        if label == self.label:
            self._label_hist.append(label)
        self._score_hist.append(score)
        self._vx_ema = 0.0
        self._vy_ema = 0.0
        self._lost_gap = 0
        # 부활 직후 얼마간 status='revived' 로 표시(프론트 상태 열).
        self._revived_ttl = max(1, int(self.config.min_hits) * 5)
        vw = max(2, int(self.config.vel_smooth_window))
        self._cxcy_hist = deque(maxlen=vw)
        _cxr, _cyr, _, _ = _to_cxcywh(bbox)
        self._cxcy_hist.append((self.age, float(_cxr), float(_cyr)))

    # ── 속성 ──────────────────────────────────────────────────────────────────

    @property
    def bbox(self):
        cx, cy, w, h = self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]
        return _to_xyxy(cx, cy, w, h)

    @property
    def label(self) -> str:
        """트랙 레이블. 고정(lock)되어 있으면 그 값을, 아니면 최근 다수결을 반환."""
        if self.config.label_lock and self._locked_label is not None:
            return self._locked_label
        return Counter(self._label_hist).most_common(1)[0][0]

    @property
    def score(self) -> float:
        return float(np.mean(self._score_hist))

    @property
    def velocity(self):
        # 화살표/로그 표시에 쓰이는 부드러운 속도(EMA).
        # 내부 칼만 상태(self.x[4],[5]) 는 그대로 유지되므로 예측 정확도엔 영향 없음.
        return (self._vx_ema, self._vy_ema)

    @property
    def status(self) -> str:
        """트랙 표시용 상태(파생값, read-only).

        기존 플래그(hit_streak·time_since_update·label lock·revive)에서 유도한
        **표시 전용** 값이며 매칭 로직에는 영향을 주지 않는다. 정식 라벨 정체성
        상태기계(전이가 매칭에 영향)는 향후 과제(docs/patent_todo.md B-3/B-5).

        - revived  : 재식별로 부활한 직후(잠시)
        - ghost    : 이번 프레임 탐지 없이 예측으로만 유지(가림)
        - locked   : 라벨이 고정(다수결 확정)된 안정 트랙
        - stable   : min_hits 이상 연속 탐지된 확정 트랙
        - tentative: 아직 확정되지 않은 신규 트랙
        """
        if self._revived_ttl > 0:
            return "revived"
        if self.time_since_update > 0:
            return "ghost"
        if self.config.label_lock and self._locked_label is not None:
            return "locked"
        if self.hit_streak >= max(1, int(self.config.min_hits)):
            return "stable"
        return "tentative"

    def mahalanobis_dist(self, bbox) -> float:
        """
        탐지 bbox와 이 트랙 예측값 사이의 마할라노비스 거리.
        칼만 공분산 P를 활용하므로 불확실성이 클 때 거리가 작아져
        새 탐지를 너그럽게 수용한다.
        """
        z = np.array(_to_cxcywh(bbox), dtype=float).reshape(4, 1)
        y = z - self.H @ self.x                  # 혁신 (residual)
        S = self.H @ self.P @ self.H.T + self.R  # 혁신 공분산
        try:
            d2 = float((y.T @ np.linalg.inv(S) @ y).item())
            dist = float(np.sqrt(max(d2, 0.0)))
        except (np.linalg.LinAlgError, ValueError):
            dist = float("inf")
        return dist


# ── 다중 객체 트래커 ──────────────────────────────────────────────────────────

class MultiObjectTracker:
    """
    카메라/시나리오별로 ``TrackerConfig`` 를 주입해 동작을 조정한다.

    사용 예
    -------
    >>> cfg = TrackerConfig(max_age=120, base_gate=13.0, vel_alpha=0.30)
    >>> tracker = MultiObjectTracker(config=cfg)

    레거시 호환을 위해 이전 키워드 인자(``max_age``, ``min_hits``,
    ``iou_threshold``, ``max_dist_px``) 도 받을 수 있다. ``config`` 와 함께
    주어지면 config 값을 override 한다.
    """

    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        *,
        max_age: Optional[int] = None,
        min_hits: Optional[int] = None,
        iou_threshold: Optional[float] = None,
        max_dist_px: Optional[float] = None,
    ):
        self.config = config or TrackerConfig()

        # 레거시 키워드 인자 → config 필드 override
        legacy = {
            "max_age":       max_age,
            "min_hits":      min_hits,
            "iou_threshold": iou_threshold,
            "max_dist_px":   max_dist_px,
        }
        for k, v in legacy.items():
            if v is not None:
                setattr(self.config, k, v)

        # 자주 쓰는 값은 속성으로도 노출 (호환성 + 가독성)
        self.max_age       = self.config.max_age
        self.min_hits      = self.config.min_hits
        self.iou_threshold = self.config.iou_threshold
        self.max_dist_px   = self.config.max_dist_px

        self.trackers: list[KalmanBoxTracker] = []
        # Re-ID 버퍼: max_age 를 넘겨 죽었지만 reid_max_age 동안 기억하는 트랙들.
        self._lost: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Parameters
        ----------
        detections : YOLO 탐지 결과 list
            [{"label": str, "score": float, "bbox": [x1,y1,x2,y2]}, ...]

        Returns
        -------
        tracks : 활성 트랙 list (is_predicted=True 포함)
        """
        self.frame_count += 1

        # ── Re-ID 버퍼 노화: 죽은 트랙 경과 +1, reid_max_age 초과는 폐기 ──────
        if self.config.reid_enabled and self._lost:
            kept = []
            for lt in self._lost:
                lt._lost_gap += 1
                if lt._lost_gap <= self.config.reid_max_age:
                    kept.append(lt)
            self._lost = kept

        # ── 0. Detection 사전 필터 ────────────────────────────────────────────
        # 가짜 박스가 트랙으로 흡수되지 않도록 트래커 진입 전에 차단.
        # - det_score_min : YOLO confidence 임계
        # - min_bbox_area : 너무 작은 박스(노이즈) 제거
        score_min = float(self.config.det_score_min)
        area_min  = float(self.config.min_bbox_area)
        if score_min > 0.0 or area_min > 0.0:
            detections = [
                d for d in detections
                if float(d.get("score", 0.0)) >= score_min
                and _bbox_area(d["bbox"]) >= area_min
            ]

        # ── 1. 칼만 예측 ──────────────────────────────────────────────────────
        pred_bboxes = [t.predict() for t in self.trackers]

        # ── 2. 탐지 결과 없을 때 ──────────────────────────────────────────────
        if not detections:
            return self._collect_tracks(set(), is_all_predicted=True)

        det_bboxes = [d["bbox"] for d in detections]
        n_trk = len(self.trackers)
        n_det = len(det_bboxes)

        # ── 3. 비용 행렬 구성 ─────────────────────────────────────────────────
        # 마할라노비스 거리(칼만 공분산 반영) + IoU 보조 게이팅 + 라벨 보너스
        # 모든 가중치는 self.config 에서 가져온다 (카메라별 조정).
        c = self.config
        base_gate              = c.base_gate
        confirmed_gate_mult    = c.confirmed_gate_mult
        label_match_bonus      = c.label_match_bonus
        label_mismatch_penalty = c.label_mismatch_penalty
        iou_bonus_factor       = c.iou_bonus_factor
        min_hits               = c.min_hits

        cost = np.ones((n_trk, n_det))
        for i, t in enumerate(self.trackers):
            gate = base_gate * (
                confirmed_gate_mult if t.hit_streak >= min_hits else 1.0
            )
            t_label = t.label

            for j, db in enumerate(det_bboxes):
                mahal = t.mahalanobis_dist(db)
                if mahal > gate:
                    continue  # 게이트 밖 → 매칭 불가

                base_cost = min(mahal / gate, 1.0)

                iou_val = _iou(pred_bboxes[i], db)
                iou_bonus = iou_val * iou_bonus_factor

                # 라벨 일치 보너스 / 불일치 페널티
                if detections[j]["label"] == t_label:
                    label_adj = -label_match_bonus
                else:
                    label_adj = +label_mismatch_penalty

                cost[i, j] = float(
                    np.clip(base_cost - iou_bonus + label_adj, 0.0, 1.0)
                )

        # ── 4. 헝가리안 알고리즘으로 최적 매칭 ──────────────────────────────
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_trk: set[int] = set()
        matched_det: set[int] = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 0.99:  # 유효한 매칭만
                self.trackers[r].update(
                    det_bboxes[c], detections[c]["label"], detections[c]["score"]
                )
                matched_trk.add(r)
                matched_det.add(c)

        # ── 4.6 IoU 2차 매칭: 마할라노비스 게이트를 벗어나 미매칭된 (고스트)트랙도
        #        detection 과 박스가 충분히 겹치면(IoU) 그 detection 으로 갱신한다.
        #        → 고스트 예측 위에 실제 탐지가 겹쳐 ID/글자가 중복 표시되는 것을 막고,
        #          기존 ID 를 유지한 채 '실제 탐지' 위치로 되돌린다.
        self._iou_fallback_match(
            detections, det_bboxes, pred_bboxes, matched_trk, matched_det
        )

        # ── 4.7 고스트 모션 재결합: 아직 살아있는(고스트) 미매칭 트랙을 모션 기반으로
        #        미매칭 detection 과 재결합한다. 가려진 사이 대상이 이동해 IoU 겹침이
        #        사라져도, '마지막 위치 + 속도 외삽' 거리로 같은 객체면 기존 ID 유지.
        #        (죽은 트랙 Re-ID 와 동일 게이팅 — 새 ID 남발 방지)
        self._ghost_motion_match(detections, det_bboxes, matched_trk, matched_det)

        # ── 4.5 Re-ID: 미매칭 detection 을 죽은 트랙(_lost)과 매칭해 ID 부활 ──
        self._reid_unmatched(detections, det_bboxes, matched_det)

        # ── 5. 미매칭 탐지 → 새 트랙 생성 ───────────────────────────────────
        for j in range(n_det):
            if j not in matched_det:
                self.trackers.append(
                    KalmanBoxTracker(
                        det_bboxes[j],
                        detections[j]["label"],
                        detections[j]["score"],
                        config=self.config,
                    )
                )

        # ── 6. 트랙 출력 + 죽은 트랙 제거 ───────────────────────────────────
        return self._collect_tracks(matched_trk)

    def _iou_fallback_match(
        self,
        detections: list[dict],
        det_bboxes: list,
        pred_bboxes: list,
        matched_trk: set,
        matched_det: set,
    ) -> None:
        """미매칭 트랙 ↔ 미매칭 detection 을 박스 겹침(IoU)으로 2차 매칭한다.

        1차(마할라노비스)에서 게이트를 벗어나 매칭되지 못한 고스트 트랙이라도,
        실제 detection 과 박스가 크게 겹치면 같은 객체로 보고 그 detection 으로
        갱신(update)한다. 그러면 ID 가 유지되고 위치가 실제 탐지로 스냅되어
        '고스트 + 새 트랙' 의 중복 표시(글자 겹침)가 사라진다.
        """
        gate = float(self.config.iou_merge_gate)
        if gate <= 0.0:
            return

        cand: list[tuple[float, int, int]] = []
        for i, _t in enumerate(self.trackers):
            if i in matched_trk:
                continue
            for j in range(len(det_bboxes)):
                if j in matched_det:
                    continue
                iou = _iou(pred_bboxes[i], det_bboxes[j])
                if iou >= gate:
                    cand.append((iou, i, j))

        # IoU 가 큰 쌍부터 그리디로 1:1 매칭.
        cand.sort(reverse=True)
        used_t: set[int] = set()
        used_d: set[int] = set()
        for _iou_val, i, j in cand:
            if i in used_t or j in used_d:
                continue
            self.trackers[i].update(
                det_bboxes[j], detections[j]["label"], detections[j]["score"]
            )
            matched_trk.add(i)
            matched_det.add(j)
            used_t.add(i)
            used_d.add(j)

    def _ghost_motion_match(
        self,
        detections: list[dict],
        det_bboxes: list,
        matched_trk: set,
        matched_det: set,
    ) -> None:
        """살아있는(고스트) 미매칭 트랙 ↔ 미매칭 detection 을 모션 기반으로 재결합한다.

        1차(마할라노비스)·IoU 겹침에서 모두 실패했더라도, 고스트로 유지 중인 트랙은
        가려진 동안 대상이 이동해 예측 위치와 실제 탐지가 떨어져 있을 수 있다. 이때
        '마지막 위치 + 속도 외삽' 으로부터의 거리가 (고스트 경과 프레임에 비례한)
        허용거리 이내이면 같은 객체로 보고 그 detection 으로 갱신 → 새 ID 생성 방지.
        게이팅은 Re-ID(_reid_unmatched)와 동일 파라미터(reid_*)를 재사용한다.
        """
        c = self.config
        if not c.reid_enabled:
            return
        ghosts = [
            i for i, t in enumerate(self.trackers)
            if i not in matched_trk and t.time_since_update > 0
        ]
        unmatched = [j for j in range(len(det_bboxes)) if j not in matched_det]
        if not ghosts or not unmatched:
            return

        cost = np.ones((len(ghosts), len(unmatched)))
        for gi, i in enumerate(ghosts):
            t = self.trackers[i]
            gap = int(t.time_since_update)
            cxL = float(t.x[0, 0]); cyL = float(t.x[1, 0])
            wL = max(float(t.x[2, 0]), 1.0); hL = max(float(t.x[3, 0]), 1.0)
            area_l = wL * hL
            vxL, vyL = t.velocity
            gcap = min(gap, max(0, int(c.reid_vel_cap)))  # 속도 외삽 상한(프레임)
            px = cxL + vxL * gcap
            py = cyL + vyL * gcap
            allowed = c.reid_dist_gate + c.reid_drift_per_frame * gap
            l_label = t.label
            for k, j in enumerate(unmatched):
                d = detections[j]
                label_match = (d["label"] == l_label)
                if c.reid_require_label and not label_match:
                    continue
                cxD, cyD, wD, hD = _to_cxcywh(d["bbox"])
                area_d = max(wD * hD, 1.0)
                sr = min(area_d / area_l, area_l / area_d)
                if sr < c.reid_size_ratio:
                    continue
                dist = ((cxD - px) ** 2 + (cyD - py) ** 2) ** 0.5
                if dist > allowed:
                    continue
                cost_val = dist / allowed
                if not label_match:
                    cost_val = min(cost_val + c.reid_label_mismatch_pen, 0.999)
                cost[gi, k] = cost_val

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, k in zip(row_ind, col_ind):
            if cost[r, k] < 1.0:
                i = ghosts[r]
                j = unmatched[k]
                self.trackers[i].update(
                    det_bboxes[j], detections[j]["label"], detections[j]["score"]
                )
                matched_trk.add(i)
                matched_det.add(j)

    def _reid_unmatched(self, detections: list[dict], det_bboxes: list, matched_det: set) -> None:
        """미매칭 detection 을 죽은 트랙(_lost)과 모션 기반으로 매칭해 ID 를 부활시킨다.

        후보 게이팅: (라벨 일치) + (면적 비율 ≥ reid_size_ratio)
                   + (마지막 위치 + 속도 외삽 으로부터의 거리 ≤ 시간에 비례한 허용거리)
        헝가리안으로 1:1 매칭 후, 통과한 쌍은 해당 트랙을 revive() 한다.
        """
        c = self.config
        if not c.reid_enabled or not self._lost:
            return
        unmatched = [j for j in range(len(det_bboxes)) if j not in matched_det]
        if not unmatched:
            return

        n_lost, n_un = len(self._lost), len(unmatched)
        cost = np.ones((n_lost, n_un))
        for i, lt in enumerate(self._lost):
            gap = lt._lost_gap
            cxL = float(lt.x[0, 0]); cyL = float(lt.x[1, 0])
            wL = max(float(lt.x[2, 0]), 1.0); hL = max(float(lt.x[3, 0]), 1.0)
            area_l = wL * hL
            vxL, vyL = lt.velocity
            gcap = min(gap, max(0, int(self.config.reid_vel_cap)))  # 속도 외삽 상한(프레임)
            px = cxL + vxL * gcap
            py = cyL + vyL * gcap
            allowed = c.reid_dist_gate + c.reid_drift_per_frame * gap
            l_label = lt.label
            for k, j in enumerate(unmatched):
                d = detections[j]
                label_match = (d["label"] == l_label)
                # require_label=False 면 라벨이 달라도(YOLO 오탐) 위치/크기로 부활 허용.
                if c.reid_require_label and not label_match:
                    continue
                cxD, cyD, wD, hD = _to_cxcywh(d["bbox"])
                area_d = max(wD * hD, 1.0)
                sr = min(area_d / area_l, area_l / area_d)
                if sr < c.reid_size_ratio:
                    continue
                dist = ((cxD - px) ** 2 + (cyD - py) ** 2) ** 0.5
                if dist > allowed:
                    continue
                cost_val = dist / allowed
                # 라벨 불일치는 허용하되 같은 라벨보다 덜 선호하도록 비용 가산.
                if not label_match:
                    cost_val = min(cost_val + c.reid_label_mismatch_pen, 0.999)
                cost[i, k] = cost_val

        row_ind, col_ind = linear_sum_assignment(cost)
        revived: set[int] = set()
        for r, k in zip(row_ind, col_ind):
            if cost[r, k] < 1.0:
                lt = self._lost[r]
                j = unmatched[k]
                lt.revive(det_bboxes[j], detections[j]["label"], detections[j]["score"])
                self.trackers.append(lt)
                matched_det.add(j)
                revived.add(r)
        if revived:
            self._lost = [lt for idx, lt in enumerate(self._lost) if idx not in revived]

    def _collect_tracks(self, matched_trk: set, is_all_predicted: bool = False) -> list[dict]:
        tracks_out = []
        survivors = []
        for i, t in enumerate(self.trackers):
            if t.time_since_update > self.max_age:
                # 고스트 수명(max_age) 초과 → 화면에서 내리되, Re-ID 용으로 기억.
                if self.config.reid_enabled:
                    t._lost_gap = 0
                    self._lost.append(t)
                continue
            if t.time_since_update <= self.max_age:
                survivors.append(t)

                # 이동 경로 갱신 (실탐지/고스트 무관하게 매 프레임 1회)
                cx_now = int(t.x[0, 0])
                cy_now = int(t.x[1, 0])
                t._pos_history.append((cx_now, cy_now))

                # 초기 노이즈 제거: min_hits 미충족 신규 트랙은 표시 안 함
                # 단, 한 번 확정된 트랙(age > min_hits)은 재탐지 시 바로 표시
                confirmed = t.age > self.min_hits
                if t.hit_streak >= self.min_hits or confirmed:
                    is_pred = is_all_predicted or (i not in matched_trk and not is_all_predicted) or t.time_since_update > 0
                    vx, vy = t.velocity
                    tracks_out.append({
                        "track_id":    t.id,
                        "label":       t.label,
                        "score":       round(t.score, 4),
                        "bbox":        t.bbox,
                        "is_predicted": bool(t.time_since_update > 0),
                        "status":      t.status,
                        "age":         t.age,
                        "hit_streak":  t.hit_streak,
                        "vx":          round(vx, 3),
                        "vy":          round(vy, 3),
                        "trail":       list(t._pos_history),
                    })
        self.trackers = survivors
        return tracks_out

    def reset(self):
        """새 영상 처리 전 상태 초기화."""
        self.trackers = []
        self._lost = []
        self.frame_count = 0
        KalmanBoxTracker._next_id = 0
