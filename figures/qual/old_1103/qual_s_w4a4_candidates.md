# 정성 후보 (s_w4a4)

규칙: FP 점수 >= 0.3이고 GT와 맞는 박스 중 Ours만 FP와 같은 클래스(BRECQ는 다른 클래스/놓침), loss 0인 이미지.
전체 대상 8103개: Ours만 맞힘 5174, BRECQ만 맞힘 1, 둘 다 맞힘 0, 둘 다 틀림 2928.

| image_id | 사례 (FP·Ours 클래스 [빈도] → BRECQ) | FP 박스 수(점수 >= t_fp) |
|---|---|---|
| 153229 | frisbee [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침; armband [c] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침 | 19 |
| 127394 | cup [f] → 놓침; cup [f] → 놓침; dining table [f] → 놓침; plate [f] → 놓침; plate [f] → 놓침; plate [f] → 놓침; beer can [c] → 놓침 | 36 |
| 568213 | frisbee [f] → 놓침; armband [c] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침 | 11 |
| 15956 | bucket [f] → 놓침; horse [f] → 놓침; wall clock [c] → 놓침; wall clock [c] → 놓침 | 15 |
| 16958 | chair [f] → 놓침; chair [f] → 놓침; chair [f] → 놓침; oil lamp [c] → 놓침 | 14 |
| 374052 | bird [f] → 놓침; pigeon [c] → 놓침; pigeon [c] → 놓침; bird [f] → 놓침 | 16 |
| 441586 | bicycle [f] → 놓침; bicycle [f] → 놓침; golfcart [c] → 놓침; bicycle [f] → 놓침 | 12 |
| 60823 | pigeon [c] → 놓침; pigeon [c] → 놓침; pigeon [c] → 놓침 | 32 |
| 292005 | shampoo [c] → 놓침; shampoo [c] → 놓침; shampoo [c] → 놓침 | 8 |
| 381971 | fireplug [f] → 놓침; horse [f] → 놓침; horse carriage [c] → 놓침 | 11 |
| 10707 | beer can [c] → 놓침; beer can [c] → 놓침 | 29 |
| 57149 | school bus [c] → 놓침; train (railroad vehicle) [f] → 놓침 | 10 |
| 156278 | wall clock [c] → 놓침; plate [f] → 놓침 | 51 |
| 229659 | wall clock [c] → 놓침; fan [f] → 놓침 | 9 |
| 265816 | horse [f] → 놓침; horse carriage [c] → 놓침 | 10 |
| 383606 | sink [f] → 놓침; water faucet [c] → 놓침 | 13 |
| 416534 | rocking chair [c] → 놓침; lamp [f] → 놓침 | 22 |
| 478393 | teddy bear [f] → 놓침; kitten [c] → 놓침 | 7 |
| 44877 | wheelchair [c] → 놓침 | 3 |
| 54628 | wedding cake [c] → 놓침 | 14 |
| 88462 | school bus [c] → 놓침 | 2 |
| 140286 | horse carriage [c] → 놓침 | 6 |
| 209747 | kitten [c] → 놓침 | 3 |
| 233139 | wall clock [c] → 놓침 | 3 |
| 330818 | wall clock [c] → 놓침 | 23 |
| 387387 | school bus [c] → 놓침 | 1 |
| 509008 | school bus [c] → 놓침 | 4 |
| 532481 | parasail (sports) [c] → 놓침 | 4 |
| 547816 | shampoo [c] → 놓침 | 8 |
| 553221 | parasail (sports) [c] → 놓침 | 8 |
| 566758 | school bus [c] → 놓침 | 7 |
| 414795 | elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침 | 39 |
| 546556 | sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침 | 39 |
| 397133 | pan (for cooking) [f] → 놓침; pan (for cooking) [f] → 놓침; pan (for cooking) [f] → 놓침; pan (for cooking) [f] → 놓침; pan (for cooking) [f] → 놓침; pan (for cooking) [f] → 놓침; pan (for cooking) [f] → 놓침; pan (for cooking) [f] → 놓침; pan (for cooking) [f] → 놓침; pan (for cooking) [f] → 놓침; pan (for cooking) [f] → 놓침; pan (for cooking) [f] → 놓침; pan (for cooking) [f] → 놓침; pan (for cooking) [f] → 놓침; pan (for cooking) [f] → 놓침; pan (for cooking) [f] → 놓침 | 56 |
| 152740 | cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침 | 20 |
| 191845 | umbrella [f] → 놓침; umbrella [f] → 놓침; umbrella [f] → 놓침; umbrella [f] → 놓침; umbrella [f] → 놓침; umbrella [f] → 놓침; umbrella [f] → 놓침; umbrella [f] → 놓침; umbrella [f] → 놓침; umbrella [f] → 놓침; umbrella [f] → 놓침; umbrella [f] → 놓침; umbrella [f] → 놓침; umbrella [f] → 놓침 | 35 |
| 238866 | doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침 | 33 |
| 245915 | elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침 | 14 |
| 548267 | sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침 | 12 |
| 173799 | elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침 | 19 |
| 559099 | cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침 | 19 |
| 254016 | doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침; doughnut [f] → 놓침 | 33 |
| 281759 | flip-flop (sandal) [f] → 놓침; flip-flop (sandal) [f] → 놓침; flip-flop (sandal) [f] → 놓침; flip-flop (sandal) [f] → 놓침; flip-flop (sandal) [f] → 놓침; flip-flop (sandal) [f] → 놓침; flip-flop (sandal) [f] → 놓침; pickup truck [f] → 놓침; flip-flop (sandal) [f] → 놓침; flip-flop (sandal) [f] → 놓침 | 23 |
| 314034 | cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; duck [f] → 놓침; cow [f] → 놓침 | 18 |
| 319935 | armchair [f] → 놓침; lamp [f] → 놓침; television set [f] → 놓침; air conditioner [f] → 놓침; pillow [f] → 놓침; lamp [f] → 놓침; pillow [f] → 놓침; pillow [f] → 놓침; pillow [f] → 놓침; pillow [f] → 놓침 | 29 |
| 361730 | kite [f] → 놓침; kite [f] → 놓침; kite [f] → 놓침; kite [f] → 놓침; kite [f] → 놓침; kite [f] → 놓침; kite [f] → 놓침; kite [f] → 놓침; kite [f] → 놓침; kite [f] → 놓침 | 32 |
| 383921 | elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침 | 11 |
| 406570 | apple [f] → 놓침; apple [f] → 놓침; apple [f] → 놓침; apple [f] → 놓침; apple [f] → 놓침; apple [f] → 놓침; apple [f] → 놓침; apple [f] → 놓침; apple [f] → 놓침; apple [f] → 놓침 | 19 |
| 435081 | cake [f] → 놓침; cake [f] → bowl; cupcake [f] → 놓침; cupcake [f] → 놓침; cupcake [f] → 놓침; cupcake [f] → 놓침; carrot [f] → 놓침; cupcake [f] → 놓침; teapot [f] → 놓침; cupcake [f] → 놓침 | 76 |
| 481390 | shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침 | 24 |
| 534394 | sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침 | 17 |
| 570688 | kite [f] → 놓침; kite [f] → 놓침; kite [f] → 놓침; kite [f] → 놓침; kite [f] → 놓침; kite [f] → 놓침; kite [f] → 놓침; kite [f] → 놓침; kite [f] → 놓침; kite [f] → 놓침 | 27 |
| 19109 | motorcycle [f] → 놓침; motorcycle [f] → 놓침; motorcycle [f] → 놓침; motorcycle [f] → 놓침; motorcycle [f] → 놓침; motorcycle [f] → 놓침; motorcycle [f] → 놓침; motorcycle [f] → 놓침; motorcycle [f] → 놓침 | 17 |
| 26564 | mouse (computer equipment) [f] → 놓침; mouse (computer equipment) [f] → 놓침; desk [f] → 놓침; computer keyboard [f] → 놓침; speaker (stero equipment) [f] → 놓침; laptop computer [f] → 놓침; speaker (stero equipment) [f] → 놓침; book [f] → 놓침; book [f] → 놓침 | 13 |
| 29640 | broccoli [f] → 놓침; broccoli [f] → 놓침; broccoli [f] → 놓침; broccoli [f] → 놓침; broccoli [f] → 놓침; broccoli [f] → 놓침; broccoli [f] → 놓침; broccoli [f] → 놓침; broccoli [f] → 놓침 | 19 |
| 68387 | helmet [f] → 놓침; baseball glove [f] → 놓침; baseball bat [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침 | 12 |
| 166391 | traffic light [f] → 놓침; traffic light [f] → 놓침; traffic light [f] → 놓침; traffic light [f] → 놓침; street sign [f] → 놓침; traffic light [f] → 놓침; traffic light [f] → 놓침; traffic light [f] → 놓침; traffic light [f] → 놓침 | 19 |
| 242946 | plate [f] → 놓침; plate [f] → 놓침; plate [f] → 놓침; necklace [f] → 놓침; plate [f] → 놓침; plate [f] → 놓침; plate [f] → 놓침; chair [f] → 놓침; chair [f] → 놓침 | 43 |
| 260470 | cake [f] → 놓침; cake [f] → 놓침; cake [f] → 놓침; cake [f] → 놓침; cake [f] → 놓침; cake [f] → 놓침; cake [f] → 놓침; cake [f] → 놓침; cake [f] → 놓침 | 31 |
| 303566 | apple [f] → 놓침; apple [f] → 놓침; apple [f] → 놓침; apple [f] → 놓침; apple [f] → 놓침; apple [f] → 놓침; apple [f] → 놓침; apple [f] → 놓침; apple [f] → 놓침 | 77 |
