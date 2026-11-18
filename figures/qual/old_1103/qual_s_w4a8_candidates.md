# 정성 후보 (s_w4a8)

규칙: FP 점수 >= 0.3이고 GT와 맞는 박스 중 Ours만 FP와 같은 클래스(BRECQ는 다른 클래스/놓침), loss 0인 이미지.
전체 대상 8104개: Ours만 맞힘 835, BRECQ만 맞힘 232, 둘 다 맞힘 6642, 둘 다 틀림 395.

| image_id | 사례 (FP·Ours 클래스 [빈도] → BRECQ) | FP 박스 수(점수 >= t_fp) |
|---|---|---|
| 374052 | pigeon [c] → bird; bird [f] → pigeon; bird [f] → pigeon | 16 |
| 343218 | racket [c] → baseball bat; tennis ball [f] → 놓침 | 11 |
| 54605 | whipped cream [c] → cupcake | 12 |
| 54628 | wedding cake [c] → birthday cake | 14 |
| 73326 | crock pot [c] → bread-bin | 18 |
| 156643 | cigarette case [c] → matchbox | 13 |
| 184611 | tricycle [c] → convertible (automobile) | 16 |
| 547816 | shampoo [c] → bottle | 8 |
| 553221 | parasail (sports) [c] → kite | 8 |
| 343561 | helmet [f] → cap (headwear); helmet [f] → cap (headwear); helmet [f] → cap (headwear); helmet [f] → cap (headwear); helmet [f] → cap (headwear); helmet [f] → cap (headwear); helmet [f] → cap (headwear); helmet [f] → cap (headwear) | 37 |
| 345469 | doughnut [f] → cupcake; doughnut [f] → bread; doughnut [f] → cake; doughnut [f] → cupcake; doughnut [f] → bread | 43 |
| 475678 | suitcase [f] → briefcase; suitcase [f] → briefcase; suitcase [f] → 놓침; suitcase [f] → briefcase; suitcase [f] → shopping bag | 15 |
| 25394 | wine bottle [f] → bottle; wine bottle [f] → bottle; cabinet [f] → cupboard; wineglass [f] → bottle | 45 |
| 83531 | sausage [f] → doughnut; sausage [f] → doughnut; sausage [f] → doughnut; sausage [f] → doughnut | 10 |
| 253742 | umbrella [f] → parasol; umbrella [f] → parasol; umbrella [f] → parasol; umbrella [f] → parasol | 29 |
| 405195 | doughnut [f] → cake; doughnut [f] → bread; doughnut [f] → 놓침; doughnut [f] → bread | 60 |
| 406570 | apple [f] → apricot; apple [f] → 놓침; apple [f] → apricot; apple [f] → 놓침 | 19 |
| 546556 | sheep [f] → shepherd dog; sheep [f] → cow; sheep [f] → 놓침; sheep [f] → 놓침 | 39 |
| 41990 | ski parka [f] → 놓침; ski parka [f] → 놓침; ski parka [f] → 놓침 | 15 |
| 103585 | vase [f] → watering can; faucet [f] → mixer (kitchen tool); mirror [f] → cupboard | 19 |
| 113720 | pizza [f] → hotplate; pizza [f] → plate; tablecloth [f] → 놓침 | 22 |
| 131138 | radiator [f] → 놓침; spoon [f] → pen; lamp [f] → 놓침 | 24 |
| 182441 | bird [f] → duck; bird [f] → duck; bird [f] → pelican | 20 |
| 276804 | helmet [f] → cap (headwear); flag [f] → 놓침; helmet [f] → 놓침 | 18 |
| 283070 | orange (fruit) [f] → lime; orange (fruit) [f] → banana; orange (fruit) [f] → lime | 6 |
| 296649 | helmet [f] → cap (headwear); helmet [f] → cap (headwear); signboard [f] → 놓침 | 21 |
| 304404 | helmet [f] → cap (headwear); baseball bat [f] → 놓침; umbrella [f] → 놓침 | 39 |
| 309391 | bicycle [f] → 놓침; traffic light [f] → 놓침; traffic light [f] → 놓침 | 17 |
| 322211 | doughnut [f] → toast (food); doughnut [f] → cupcake; doughnut [f] → cupcake | 12 |
| 384808 | sink [f] → washbasin; bottle [f] → shampoo; toilet [f] → 놓침 | 16 |
| 440475 | vase [f] → hatbox; candle [f] → cup; book [f] → mousepad | 43 |
| 477288 | boot [f] → 놓침; boot [f] → 놓침; boot [f] → 놓침 | 14 |
| 547336 | fireplug [f] → ottoman; pencil [f] → nailfile; sweater [f] → person | 10 |
| 578500 | armchair [f] → sofa bed; vase [f] → 놓침; vase [f] → 놓침 | 28 |
| 9400 | computer keyboard [f] → phonebook; bottle [f] → 놓침 | 19 |
| 33104 | helmet [f] → cap (headwear); helmet [f] → cap (headwear) | 13 |
| 44699 | sheep [f] → shepherd dog; sheep [f] → shepherd dog | 5 |
| 60507 | helmet [f] → cap (headwear); helmet [f] → cap (headwear) | 18 |
| 64523 | tennis racket [f] → racket; tennis racket [f] → 놓침 | 10 |
| 65074 | headboard [f] → cupboard; pillow [f] → cushion | 5 |
| 91406 | pizza [f] → hotplate; pizza [f] → 놓침 | 21 |
| 92939 | watch [f] → armband; flowerpot [f] → ottoman | 15 |
| 107339 | coffee table [f] → ottoman; coffee table [f] → ottoman | 9 |
| 109118 | boot [f] → 놓침; boot [f] → 놓침 | 7 |
| 116208 | pizza [f] → toast (food); pizza [f] → 놓침 | 22 |
| 123321 | broccoli [f] → 놓침; broccoli [f] → 놓침 | 4 |
| 132544 | helmet [f] → cap (headwear); helmet [f] → cap (headwear) | 20 |
| 136715 | helmet [f] → cap (headwear); boot [f] → 놓침 | 19 |
| 136772 | strawberry [f] → tomato; strawberry [f] → tomato | 9 |
| 137576 | cow [f] → 놓침; cow [f] → shepherd dog | 9 |
| 138954 | cucumber [f] → lime; cucumber [f] → 놓침 | 13 |
| 149406 | wheel [f] → 놓침; wheel [f] → 놓침 | 5 |
| 154705 | laptop computer [f] → monitor (computer equipment) computer monitor; printer [f] → briefcase | 20 |
| 158660 | apple [f] → tomato; banana [f] → hamburger | 14 |
| 163117 | kite [f] → pelican; kite [f] → 놓침 | 21 |
| 165039 | traffic light [f] → 놓침; traffic light [f] → 놓침 | 15 |
| 167898 | mirror [f] → cupboard; bathtub [f] → 놓침 | 10 |
| 170670 | beer bottle [f] → bottle; beer bottle [f] → bottle | 22 |
| 171190 | wine bottle [f] → bottle; knife [f] → tongs | 28 |
| 173799 | elephant [f] → sheep; elephant [f] → 놓침 | 19 |
