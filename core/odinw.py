"""ODinW-13 평가 (10-27). COCO/LVIS 밖의 데이터에서 일반화를 확인하려고 추가했다.

데이터는 GLIP 공식 배포본(HuggingFace GLIPModel/GLIP의 odinw_35/)을 /data/taeho/odinw에 받아 두었다
(/data/taeho/odinw/download_odinw13.sh). 평가 split과 프롬프트는 GLIP의 configs/odinw_13/*.yaml을 그대로 따른다
(같은 폴더의 glip_configs_odinw13/에 저장):
  - split: DATASETS.TEST가 가리키는 것. 대부분 "val", EgoHands와 VehiclesOpenImages는 "minival"("test"가 아니다).
  - 프롬프트: DATASETS.OVERRIDE_CATEGORY의 이름. 13개 모두 어노테이션의 categories 이름과 같다("CoW" 등 포함).
지표: 데이터셋마다 COCO AP(pycocotools, maxDets 100), 주 지표는 13개 AP의 단순 평균(GLIP 관행).
데이터셋 크기가 4장~3422장으로 크게 달라 작은 데이터셋이 평균을 흔들 수 있으므로 데이터셋별 값도 같이 남긴다.
"""
from __future__ import annotations
import contextlib, glob, io, json, os

ODINW_ROOT = "/data/taeho/odinw"


def load_odinw13(root=ODINW_ROOT, cap=0):
    """GLIP config 13개를 읽어 [dict(name, img_paths, img_ids, names, cat_ids, gt)]를 돌려준다.
    cap>0이면 데이터셋마다 앞에서 cap장만(빠른 점검용)."""
    import yaml
    from pycocotools.coco import COCO
    cfgs = sorted(glob.glob(os.path.join(root, "glip_configs_odinw13", "*.yaml")))
    assert len(cfgs) == 13, f"GLIP ODinW-13 config가 {len(cfgs)}개 ({root}/glip_configs_odinw13)"
    sets = []
    for f in cfgs:
        D = yaml.safe_load(open(f))["DATASETS"]
        split = D["TEST"].strip("()").replace('"', "").split(",")[0].strip()
        reg = D["REGISTER"][split]
        ann = os.path.join(root, reg["ann_file"].split("/", 1)[1])          # "odinw/..." -> root/...
        img_dir = os.path.join(root, reg["img_dir"].split("/", 1)[1])
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(ann)
            # ODinW 어노테이션 id는 0부터 시작하는데, COCOeval은 매칭 기록(gtm/dtm)에서 id 0을 "매칭 안 됨"으로 본다.
            # 그대로 두면 id 0 GT에 맞은 검출이 오탐이 된다(GT를 그대로 넣어도 CottontailRabbits AP 90.5). 1부터 다시 매긴다.
            for j, a in enumerate(gt.dataset["annotations"]):
                a["id"] = j + 1
            gt.createIndex()
        cats = sorted(json.loads(D["OVERRIDE_CATEGORY"]), key=lambda c: c["id"])
        assert [c["id"] for c in cats] == sorted(gt.getCatIds()), f"{f}: 클래스 id가 어노테이션과 다르다"
        imgs = sorted(gt.dataset["images"], key=lambda i: i["id"])
        if cap > 0:
            imgs = imgs[:cap]
        sets.append(dict(name=reg["ann_file"].split("/")[1], split=split,
                         img_paths=[os.path.join(img_dir, i["file_name"]) for i in imgs],
                         img_ids=[i["id"] for i in imgs],
                         names=[c["name"] for c in cats], cat_ids=[c["id"] for c in cats], gt=gt))
    return sets


def coco_ap(gt, results, img_ids):
    """pycocotools bbox AP@[.5:.95]. 검출이 없으면 0."""
    from pycocotools.cocoeval import COCOeval
    if not results:
        return 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(results), "bbox")
        ev.params.imgIds = list(img_ids)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0])


def report(res_by_model, sets):
    """{모델: {데이터셋: AP(%)}} -> 각 모델에 'mean' 키를 더하고 표를 출력한다.
    로그 파싱용으로 모델마다 '[ODinW-AP][<모델>] mean=... | <데이터셋>=...' 한 줄을 남긴다."""
    names = [s["name"] for s in sets]
    for k, r in res_by_model.items():
        r["mean"] = sum(r[n] for n in names) / len(names)
    print("\n" + "=" * 100)
    print(f" ODinW-13 AP (GLIP 공식 split, 데이터셋 AP 단순 평균) -- 이미지 수: "
          + ", ".join(f"{s['name']} {len(s['img_ids'])}" for s in sets))
    print("=" * 100)
    for k, r in res_by_model.items():
        print(f"[ODinW-AP][{k}] mean={r['mean']:.2f} | " + " ".join(f"{n}={r[n]:.2f}" for n in names))
    return res_by_model
