# Ours만 맞힌 사례 상위 10장 (s_w4a5)

규칙: FP 점수 >= 0.3이고 GT와 맞는 박스 중 Ours만 FP와 같은 클래스(BRECQ는 다른 클래스/놓침), loss 0인 이미지.
전체 대상 8103개: Ours만 맞힘 1724, BRECQ만 맞힘 315, 둘 다 맞힘 5272, 둘 다 틀림 792.

| 파일 | image_id | 사례 (FP·Ours 클래스 [빈도] → BRECQ) |
|---|---|---|
| 01_60823.png | 60823 | cow [f] → bull; pigeon [c] → 놓침; pigeon [c] → 놓침; pigeon [c] → 놓침; pigeon [c] → 놓침; pigeon [c] → 놓침; pigeon [c] → 놓침; pigeon [c] → 놓침; pigeon [c] → 놓침 |
| 02_104669.png | 104669 | carrot [f] → 놓침; steak (food) [c] → toast (food); potato [f] → cauliflower |
| 03_184611.png | 184611 | tricycle [c] → motor scooter; tricycle [c] → golfcart; signboard [f] → 놓침 |
| 04_374052.png | 374052 | pigeon [c] → bird; pigeon [c] → 놓침; pigeon [c] → 놓침 |
| 05_57149.png | 57149 | school bus [c] → 놓침; streetlight [f] → 놓침 |
| 06_187734.png | 187734 | ladder [f] → 놓침; racket [c] → tennis racket |
| 07_292005.png | 292005 | shampoo [c] → dishwasher detergent; shampoo [c] → bottle |
| 08_343218.png | 343218 | racket [c] → 놓침; tennis ball [f] → 놓침 |
| 09_410880.png | 410880 | teddy bear [f] → shepherd dog; lantern [c] → 놓침 |
| 10_443969.png | 443969 | skirt [f] → 놓침; shopping cart [c] → 놓침 |
