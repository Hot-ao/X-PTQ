"""
Combined(PromptCal) 기준선(promptcal.py)이 쓰는 도우미.

  _LevelCapture            : cv2/cv3/cv4 등 레벨별 ModuleList의 출력을 캡처해 [anchors, C]로 조립.
  get_txt_feats            : 모델에 설정된 텍스트 임베딩.
  text_neighbor_order      : 텍스트 임베딩 유사도 기준의 프롬프트별 이웃 순서(neighbor 항의 후보).
  utility_refinement_terms : threshold-crossing hinge와 box consistency 항. 확정 프로토콜에서는
                             꺼져 있다(optimize_promptcal_scale_neighbor의 utility_stage2_frac=0).

원래 이 파일에 있던 Semantic Calibration 최적화(optimize_semantic_pcal)는 기각된 방향이라 제거했다
(옛 구현은 promptcal-ptq의 src/quant/semantic_calib.py, 경위는 PROMPTCAL_HISTORY).
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


class _LevelCapture:
    """cv2/cv3/cv4 등 레벨별 ModuleList의 출력을 캡처해 [anchors, C]로 조립."""

    def __init__(self, module_list):
        self.buf = {}
        self.handles = []
        for i, sub in enumerate(module_list):
            def make(idx):
                def hook(_m, _inp, out):
                    self.buf[idx] = out
                return hook
            self.handles.append(sub.register_forward_hook(make(i)))

    def clear(self):
        self.buf = {}

    def assemble(self):
        parts = []
        for i in sorted(self.buf):
            t = self.buf[i]
            B, C, H, W = t.shape
            parts.append(t.reshape(B, C, H * W))
        cat = torch.cat(parts, dim=2)                 # [B, C, A]
        return cat[0].transpose(0, 1).contiguous()     # [A, C]

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def get_txt_feats(model_module):
    """WorldModel.txt_feats: [1, P, 512], L2-normalized (fuse 전/후 모두 존재).
    이건 FP text encoder 출력이라 양자화 대상이 아니고, 항상 FP 모델에서 뽑는다."""
    return model_module.txt_feats[0].detach()


def text_neighbor_order(txt_feats):
    """txt_feats: [P, 512] (L2-normalized). 반환: [P, P] 각 class별 유사도 내림차순
    인덱스(자기 자신은 맨 뒤로 밀림)."""
    sim = txt_feats @ txt_feats.T
    sim = sim.clone()
    sim.fill_diagonal_(-2.0)
    return sim.argsort(dim=-1, descending=True)


def utility_refinement_terms(sim_q, sim_fp, cv2_q, cv2_fp, pidx, det_thres=0.25,
                             conf_thres=0.25, margin_thres=0.5):
    """
    Utility Constraints: threshold-crossing + box consistency.
    reliable anchor(§semantic_calibration_loss와 동일 기준)에서만 계산.
    """
    device = sim_q.device
    sub_fp = sim_fp[:, pidx]
    prob_fp = sub_fp.sigmoid()
    top2 = sub_fp.topk(min(2, sub_fp.shape[-1]), dim=-1).values
    maxp = prob_fp.max(-1).values
    margin = top2[:, 0] - top2[:, 1] if top2.shape[-1] > 1 else top2[:, 0]
    reliable = (maxp > conf_thres) & (margin > margin_thres)
    if reliable.sum() == 0:
        z = sim_q.sum() * 0.0
        return z, z

    aidx = reliable.nonzero(as_tuple=True)[0]
    local_argmax = sub_fp[aidx].argmax(-1)
    targets = pidx[local_argmax]

    # threshold crossing: FP가 det_thres 위였던 target column의 quant 확률이
    # 그 밑으로 떨어지지 않도록 one-sided hinge.
    #
    # 09-25 버그 수정: prob_fp는 **전체 anchor** 배열인데 행 인덱스를
    # torch.arange(len(aidx))로 주고 있었다 -- 선택된 reliable anchor(aidx)가 아니라
    # 앞에서부터 len(aidx)개 행(대부분 배경 anchor)을 보게 된다. 같은 함수의 다른
    # 줄(sub_fp[aidx], sim_q[aidx, targets], cv2_q[aidx])은 전부 aidx를 제대로 쓴다.
    # 실측 결과 reliable anchor 538개 중 fp_positive가 **0개**라 l_thresh가 항상
    # 정확히 0이었다 -- 즉 논문 §4.3의 threshold-crossing 항이 한 번도 작동한 적이
    # 없다(이 helper를 쓰는 optimize_semantic_pcal(30번)과 구
    # optimize_promptcal_scale_neighbor_utility도 마찬가지). thresh_w를 1.0/2.0으로
    # 바꿔도 결과가 bit-identical하게 나온 것으로 확인됨(runs/103).
    # 올바른 인덱싱은 prob_fp[aidx, local_argmax] (= maxp[aidx]).
    q_prob_t = sim_q[aidx, targets].sigmoid()
    fp_positive = prob_fp[aidx, local_argmax] > det_thres
    if fp_positive.any():
        l_thresh = F.relu(det_thres - q_prob_t[fp_positive]).pow(2).mean()
    else:
        l_thresh = sim_q.sum() * 0.0

    # box consistency: reliable anchor에서 cv2(box regression 원시 출력) MSE.
    l_box = F.mse_loss(cv2_q[aidx], cv2_fp[aidx])
    return l_thresh, l_box
