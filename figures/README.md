# figures (11-17 정리)

그림 출력은 역할별 폴더에, 만드는 코드는 모두 `src/`에 있다(concept_flip만 자체 `src/`).
스크립트는 저장소 루트에서 `.venv/bin/python figures/src/<script>.py`로 돌리면 아래 폴더에 바로 저장한다.

| 폴더 | 그림 | 코드 (src/) |
|---|---|---|
| `teaser/` | teaser_flip, teaser_v2, teaser_v3, teaser_v4 | teaser_*.py |
| `concept_flip/` | Fig.1 개념도(voronoi), concept_A/B/C, pptx | concept_flip/src/ |
| `method/` | method_arch, method_flow(+pptx), method_overview | make_method_*.py |
| `analysis/` | concentration_curve, sensitivity_maps(_all), diag_vocab_scatter, mcheck_scatter | concentration_curve.py, diag_concentration.py, diag_vocab_scatter.py, mcheck_scatter.py |
| `experiments/` | bits_curve, vocab_scale(flip), vocab_scale_ap(2x2), vocab_scale_combo, vocab_scale_one(막대+선 시안) | bits_curve.py, vocab_scale*_fig.py |
| `qual/` | qual_grid(부록 비트별 정성), qual_s_w4a8 + 후보 목록 | qual_grid.py, qual_figure.py, qual_dets.py, make_qual_grid_pptx.py |
| `qual/old_1103/` | 11-03 정성 그림 이전판(qual_grid, qual_s_w4a{4,5,8}, *_wins) | - |
| `reference/` | YOLO-World 원 논문 그림(참고용) | - |

논문(Overleaf) 쪽 그림은 archive/의 각 Overleaf 폴더에 따로 복사돼 있어 이 정리와 무관하다.
