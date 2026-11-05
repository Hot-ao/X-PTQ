# 정성 후보 (s_w4a5)

규칙: FP 점수 >= 0.3이고 GT와 맞는 박스 중 Ours만 FP와 같은 클래스(BRECQ는 다른 클래스/놓침), loss 0인 이미지.
전체 대상 8103개: Ours만 맞힘 1724, BRECQ만 맞힘 315, 둘 다 맞힘 5272, 둘 다 틀림 792.

| image_id | 사례 (FP·Ours 클래스 [빈도] → BRECQ) | FP 박스 수(점수 >= t_fp) |
|---|---|---|
| 60823 | cow [f] → bull; pigeon [c] → 놓침; pigeon [c] → 놓침; pigeon [c] → 놓침; pigeon [c] → 놓침; pigeon [c] → 놓침; pigeon [c] → 놓침; pigeon [c] → 놓침; pigeon [c] → 놓침 | 32 |
| 104669 | carrot [f] → 놓침; steak (food) [c] → toast (food); potato [f] → cauliflower | 18 |
| 184611 | tricycle [c] → motor scooter; tricycle [c] → golfcart; signboard [f] → 놓침 | 16 |
| 374052 | pigeon [c] → bird; pigeon [c] → 놓침; pigeon [c] → 놓침 | 16 |
| 57149 | school bus [c] → 놓침; streetlight [f] → 놓침 | 10 |
| 187734 | ladder [f] → 놓침; racket [c] → tennis racket | 22 |
| 292005 | shampoo [c] → dishwasher detergent; shampoo [c] → bottle | 8 |
| 343218 | racket [c] → 놓침; tennis ball [f] → 놓침 | 11 |
| 410880 | teddy bear [f] → shepherd dog; lantern [c] → 놓침 | 8 |
| 443969 | skirt [f] → 놓침; shopping cart [c] → 놓침 | 13 |
| 23230 | goose [c] → pelican | 3 |
| 23899 | projector [c] → briefcase | 5 |
| 44877 | wheelchair [c] → bicycle | 3 |
| 54605 | whipped cream [c] → cupcake | 12 |
| 54628 | wedding cake [c] → birthday cake | 14 |
| 112110 | briefcase [c] → iPod | 4 |
| 284445 | walking cane [c] → 놓침 | 17 |
| 387387 | school bus [c] → train (railroad vehicle) | 1 |
| 393569 | bunk bed [c] → cupboard | 7 |
| 523175 | soupspoon [c] → salad plate | 2 |
| 547816 | shampoo [c] → bottle | 8 |
| 389566 | sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → 놓침 | 15 |
| 406570 | apple [f] → apricot; apple [f] → apricot; apple [f] → apricot; apple [f] → apricot; apple [f] → 놓침; apple [f] → apricot; apple [f] → apricot; apple [f] → apricot; apple [f] → 놓침 | 19 |
| 15278 | broccoli [f] → 놓침; broccoli [f] → 놓침; broccoli [f] → 놓침; broccoli [f] → 놓침; broccoli [f] → 놓침; broccoli [f] → 놓침; broccoli [f] → 놓침 | 10 |
| 41990 | ski parka [f] → 놓침; ski parka [f] → 놓침; glove [f] → 놓침; glove [f] → 놓침; ski boot [f] → 놓침; ski parka [f] → 놓침; ski pole [f] → 놓침 | 15 |
| 157390 | carrot [f] → 놓침; carrot [f] → 놓침; carrot [f] → 놓침; carrot [f] → bell pepper; carrot [f] → 놓침; carrot [f] → 놓침; carrot [f] → 놓침 | 27 |
| 238866 | doughnut [f] → cupcake; doughnut [f] → cupcake; doughnut [f] → cupcake; doughnut [f] → 놓침; doughnut [f] → crumb; doughnut [f] → cupcake; doughnut [f] → 놓침 | 33 |
| 479596 | apple [f] → 놓침; apple [f] → apricot; apple [f] → 놓침; apple [f] → apricot; apple [f] → persimmon; apple [f] → 놓침; apple [f] → 놓침 | 35 |
| 191845 | umbrella [f] → 놓침; umbrella [f] → parasol; umbrella [f] → tablecloth; umbrella [f] → 놓침; umbrella [f] → parasol; umbrella [f] → 놓침 | 35 |
| 194940 | carrot [f] → 놓침; carrot [f] → 놓침; carrot [f] → sweet potato; carrot [f] → sweet potato; carrot [f] → 놓침; carrot [f] → 놓침 | 17 |
| 309391 | traffic light [f] → 놓침; traffic light [f] → 놓침; traffic light [f] → 놓침; bicycle [f] → 놓침; traffic light [f] → 놓침; traffic light [f] → 놓침 | 17 |
| 314034 | cow [f] → shepherd dog; cow [f] → bull; duck [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침 | 18 |
| 314182 | carrot [f] → sweet potato; carrot [f] → sweet potato; carrot [f] → sweet potato; bowl [f] → soup bowl; carrot [f] → tomato; broccoli [f] → 놓침 | 14 |
| 103548 | sheep [f] → 놓침; sheep [f] → 놓침; sheep [f] → zebra; sheep [f] → 놓침; sheep [f] → 놓침 | 12 |
| 129054 | zebra [f] → 놓침; zebra [f] → 놓침; zebra [f] → 놓침; zebra [f] → 놓침; zebra [f] → 놓침 | 12 |
| 137576 | cow [f] → 놓침; cow [f] → dog; cow [f] → 놓침; cow [f] → 놓침; cow [f] → 놓침 | 9 |
| 197388 | glove [f] → 놓침; ski boot [f] → shoe; ski boot [f] → 놓침; ski boot [f] → 놓침; glove [f] → 놓침 | 19 |
| 303713 | chair [f] → 놓침; chair [f] → 놓침; chair [f] → 놓침; chair [f] → 놓침; chair [f] → 놓침 | 29 |
| 334767 | ski pole [f] → 놓침; ski boot [f] → 놓침; ski boot [f] → 놓침; ski boot [f] → shoe; ski boot [f] → 놓침 | 11 |
| 357816 | belt [f] → 놓침; umbrella [f] → 놓침; sunglasses [f] → 놓침; sunglasses [f] → 놓침; baseball [f] → 놓침 | 22 |
| 397133 | pan (for cooking) [f] → 놓침; pan (for cooking) [f] → saucepan; pan (for cooking) [f] → 놓침; pan (for cooking) [f] → saucepan; pan (for cooking) [f] → 놓침 | 56 |
| 414795 | elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침 | 39 |
| 416104 | chair [f] → 놓침; watch [f] → 놓침; spectacles [f] → 놓침; umbrella [f] → 놓침; chair [f] → 놓침 | 25 |
| 481390 | shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침; shoe [f] → 놓침 | 24 |
| 29640 | broccoli [f] → 놓침; broccoli [f] → 놓침; broccoli [f] → 놓침; broccoli [f] → 놓침 | 19 |
| 103585 | mirror [f] → cupboard; vase [f] → flower arrangement; faucet [f] → water faucet; mirror [f] → pegboard | 19 |
| 104612 | broccoli [f] → 놓침; broccoli [f] → 놓침; broccoli [f] → 놓침; broccoli [f] → 놓침 | 10 |
| 125062 | teddy bear [f] → 놓침; teddy bear [f] → 놓침; teddy bear [f] → 놓침; teddy bear [f] → 놓침 | 8 |
| 173799 | elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침; elephant [f] → 놓침 | 19 |
| 208423 | kite [f] → weathervane; kite [f] → 놓침; kite [f] → 놓침; kite [f] → 놓침 | 16 |
| 224093 | cow [f] → shepherd dog; cow [f] → elephant; cow [f] → 놓침; cow [f] → shepherd dog | 9 |
| 301421 | speaker (stero equipment) [f] → can; mousepad [f] → 놓침; speaker (stero equipment) [f] → iPod; speaker (stero equipment) [f] → 놓침 | 17 |
| 325838 | laptop computer [f] → mousepad; computer keyboard [f] → mousepad; chair [f] → cushion; desk [f] → coffee table | 14 |
| 335427 | broccoli [f] → 놓침; broccoli [f] → 놓침; broccoli [f] → 놓침; broccoli [f] → 놓침 | 11 |
| 343561 | helmet [f] → cap (headwear); helmet [f] → cap (headwear); helmet [f] → cap (headwear); helmet [f] → cap (headwear) | 37 |
| 349837 | refrigerator [f] → cupboard; refrigerator [f] → cupboard; refrigerator [f] → cupboard; refrigerator [f] → cupboard | 7 |
| 357888 | vase [f] → 놓침; mug [f] → 놓침; watch [f] → wristband; mug [f] → 놓침 | 24 |
| 373353 | boot [f] → 놓침; minivan [f] → jeep; boot [f] → 놓침; boot [f] → 놓침 | 20 |
| 378453 | zebra [f] → 놓침; zebra [f] → 놓침; zebra [f] → 놓침; zebra [f] → 놓침 | 8 |
| 410650 | chair [f] → 놓침; chair [f] → 놓침; chair [f] → 놓침; chair [f] → 놓침 | 35 |
