#!/usr/bin/env python3
"""Tự kiểm file xuất COCO Keypoints 1.0 của Lab Ngày 4 (Keypoint & Pose).

Cách chạy (từ thư mục gốc repo, Python 3.9+, không cần cài thư viện):

    python src/check_submission.py --persons src/exports/HoVaTen_persons.zip \
                                   --faces   src/exports/HoVaTen_faces.zip

Script KHÔNG có ground truth: không tính OKS, không biết điểm có nằm đúng khớp
hay không. Nó kiểm định dạng file và các dấu hiệu lỗi hay gặp. Không có cảnh
báo chưa chắc là đúng: vẫn phải tự review (CP3).

Mã thoát: 0 nếu không có [LỖI], 1 nếu còn [LỖI], 2 nếu không đọc được file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent / "keypoint_schema.json"
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
MIN_PERSON_HEIGHT = 60  # px, quy ước của lab
EDGE_MARGIN = 1.0  # px: điểm v > 0 cách mép ảnh <= 1 px bị coi là "kéo về mép"
LR_THRESHOLD = 0.12  # tỉ lệ so với cỡ thân / cỡ mặt để coi một cặp là rõ bên


def setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.v0_with_xy = 0

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        print(f"  [LỖI] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  [CẢNH BÁO] {msg}")


# ---------------------------------------------------------------- đọc file


def load_docs(path: Path, rep: Report) -> list[tuple[str, dict]]:
    """Trả về danh sách (tên file json, nội dung) từ một zip hoặc một file .json."""
    if not path.exists():
        print(f"  Không tìm thấy file: {path}")
        sys.exit(2)
    docs: list[tuple[str, dict]] = []
    if path.suffix.lower() == ".json":
        try:
            docs.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  Không đọc được JSON {path}: {exc}")
            sys.exit(2)
        return docs
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        print(f"  {path} không phải file zip hợp lệ.")
        sys.exit(2)
    with zf:
        names = zf.namelist()
        images = [n for n in names if n.lower().endswith(IMAGE_EXT)]
        if images:
            rep.error(
                f"Zip có kèm {len(images)} ảnh (ví dụ {images[0]}). "
                "Xuất lại và BỎ CHỌN 'Save images'."
            )
        for name in names:
            if not name.lower().endswith(".json"):
                continue
            try:
                docs.append((name, json.loads(zf.read(name).decode("utf-8"))))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                rep.error(f"Không đọc được {name}: {exc}")
    docs = [(n, d) for n, d in docs if isinstance(d, dict) and "annotations" in d]
    if not docs:
        rep.error(
            "Không thấy file COCO nào trong zip (cần annotations/person_keypoints_default.json). "
            "Kiểm lại định dạng xuất: COCO Keypoints 1.0."
        )
    return docs


# ---------------------------------------------------------------- hình học


def sub(p, q):
    return (p[0] - q[0], p[1] - q[1])


def dot(p, q):
    return p[0] * q[0] + p[1] * q[1]


def norm(p):
    return math.hypot(p[0], p[1])


def mean(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def lateral_unit(axis):
    """Vector vuông góc với trục thân (trục hướng từ trên xuống), quy ước sao cho
    với trục thẳng đứng (0, 1) thì vector này là (1, 0): sang phải ảnh."""
    length = norm(axis)
    if length < 1e-6:
        return None
    return (axis[1] / length, -axis[0] / length)


# ---------------------------------------------------------------- kiểm từng phần


def check_category(cat: dict, schema: dict, rep: Report, where: str) -> dict[str, int] | None:
    """Kiểm tên điểm của category. Trả về map tên -> vị trí trong mảng keypoints."""
    name = cat.get("name")
    names = cat.get("keypoints") or []
    expected = schema[name]["keypoints"]
    missing = [n for n in expected if n not in names]
    extra = [n for n in names if n not in expected]
    if missing or extra:
        rep.error(
            f"{where}: label `{name}` sai tên điểm. Thiếu: {missing or 'không'}; "
            f"thừa / sai chính tả: {extra or 'không'}. Báo coach (label do coach dựng)."
        )
        return None
    if len(names) != len(set(names)):
        rep.error(f"{where}: label `{name}` có tên điểm bị lặp.")
        return None
    if list(names) != list(expected):
        rep.warn(
            f"{where}: label `{name}` đủ tên nhưng sai thứ tự so với bảng của lab. "
            "Script vẫn đọc theo tên; báo coach để sửa label."
        )
    edges = cat.get("skeleton") or []
    try:
        got = {frozenset((names[a - 1], names[b - 1])) for a, b in edges}
        exp = {frozenset((expected[a - 1], expected[b - 1])) for a, b in schema[name]["skeleton"]}
        if got != exp:
            rep.warn(
                f"{where}: label `{name}` có {len(got)} cạnh, khác bộ {len(exp)} cạnh của lab. "
                "Cạnh chỉ để nhìn, không ảnh hưởng tọa độ; báo coach."
            )
    except (IndexError, TypeError, ValueError):
        rep.warn(f"{where}: không đọc được danh sách cạnh của label `{name}`.")
    return {n: i for i, n in enumerate(names)}


def parse_points(ann, index, schema_names, rep, tag):
    """Đọc mảng keypoints, kiểm định dạng. Trả về dict tên -> (x, y, v)."""
    kps = ann.get("keypoints")
    k = len(index)
    if not isinstance(kps, list) or len(kps) != 3 * k:
        n = len(kps) if isinstance(kps, list) else "không có"
        rep.error(f"{tag}: mảng keypoints có {n} số, cần {3 * k}.")
        return None
    pts = {}
    ok = True
    for name in schema_names:
        i = index[name]
        x, y, v = kps[3 * i], kps[3 * i + 1], kps[3 * i + 2]
        if v not in (0, 1, 2):
            rep.error(f"{tag}: `{name}` có v = {v}, chỉ được 0, 1 hoặc 2.")
            ok = False
            continue
        if v == 0 and (x != 0 or y != 0):
            # CVAT 2.x giữ tọa độ cũ cho điểm outside (chuẩn COCO ghi 0, 0).
            # Điểm v = 0 bị bỏ qua khi huấn luyện và khi chấm nên không tính là lỗi.
            rep.v0_with_xy += 1
        pts[name] = (float(x), float(y), int(v))
    if not ok:
        return pts
    nk = sum(1 for p in pts.values() if p[2] > 0)
    if ann.get("num_keypoints") != nk:
        rep.error(f"{tag}: num_keypoints = {ann.get('num_keypoints')}, nhưng có {nk} điểm v > 0.")
    return pts


def check_bounds(pts, img, rep, tag):
    w, h = img.get("width"), img.get("height")
    if not w or not h:
        return
    for name, (x, y, v) in pts.items():
        if v == 0:
            continue
        if x < 0 or y < 0 or x > w or y > h:
            rep.error(f"{tag}: `{name}` v = {v} nằm ngoài ảnh ({x:.0f}, {y:.0f}); điểm ngoài ảnh phải là outside.")
        elif x <= EDGE_MARGIN or y <= EDGE_MARGIN or x >= w - EDGE_MARGIN or y >= h - EDGE_MARGIN:
            rep.warn(
                f"{tag}: `{name}` v = {v} nằm sát mép ảnh ({x:.0f}, {y:.0f}). "
                "Nếu khớp thật ở ngoài ảnh, bật outside thay vì kéo về mép."
            )


def visible(pts, name):
    p = pts.get(name)
    return p is not None and p[2] > 0


def xy(pts, name):
    p = pts[name]
    return (p[0], p[1])


def side_signs(pts, pairs, origin_axis, scale):
    """Với mỗi cặp (left, right) đều v > 0, tính left nằm phía nào so với trục.
    Trả về list (tên cặp, dấu) chỉ với cặp lệch rõ (> LR_THRESHOLD * scale)."""
    n = lateral_unit(origin_axis)
    if n is None or scale <= 0:
        return []
    out = []
    for left, right in pairs:
        if not (visible(pts, left) and visible(pts, right)):
            continue
        p = dot(sub(xy(pts, left), xy(pts, right)), n)
        if abs(p) >= LR_THRESHOLD * scale:
            out.append((left.replace("left_", ""), 1 if p > 0 else -1))
    return out


def check_person_lr(pts, schema, rep, tag):
    have_sh = visible(pts, "left_shoulder") and visible(pts, "right_shoulder")
    have_hip = visible(pts, "left_hip") and visible(pts, "right_hip")
    if have_sh and have_hip:
        top = mean([xy(pts, "left_shoulder"), xy(pts, "right_shoulder")])
        bot = mean([xy(pts, "left_hip"), xy(pts, "right_hip")])
        axis = sub(bot, top)
        scale = max(norm(axis), norm(sub(xy(pts, "left_shoulder"), xy(pts, "right_shoulder"))))
    elif have_sh or have_hip:
        a, b = ("left_shoulder", "right_shoulder") if have_sh else ("left_hip", "right_hip")
        axis = (0.0, 1.0)
        scale = 2.5 * norm(sub(xy(pts, a), xy(pts, b)))
    else:
        return
    signs = side_signs(pts, schema["flip_pairs"], axis, scale)
    if len(signs) < 3:
        return
    pos = [s for s, v in signs if v > 0]
    neg = [s for s, v in signs if v < 0]
    if pos and neg:
        minority = pos if len(pos) < len(neg) else neg
        if len(pos) == len(neg):
            rep.warn(f"{tag}: trái/phải không nhất quán giữa các cặp điểm ({', '.join(pos)} so với {', '.join(neg)}). Kiểm `lr_swap`.")
        else:
            rep.warn(
                f"{tag}: cặp {', '.join(minority)} nằm ngược phía so với các cặp còn lại. "
                "Có thể đổi trái/phải một cặp (`lr_swap`), hoặc tay / chân bắt chéo thật: kiểm lại."
            )
        return
    face_seen = all(pts.get(n, (0, 0, 0))[2] == 2 for n in ("nose", "left_eye", "right_eye"))
    if face_seen and neg:
        rep.warn(
            f"{tag}: thấy rõ mũi và hai mắt (v = 2) nhưng các điểm left nằm bên TRÁI ảnh, "
            "tức là gán như người quay lưng. Kiểm lại hướng người (`lr_swap` cả người?)."
        )


def check_face(pts, rep, tag):
    eye_l = [xy(pts, n) for n in ("left_eye_outer", "left_eye_inner", "left_eye_upper", "left_eye_lower") if visible(pts, n)]
    eye_r = [xy(pts, n) for n in ("right_eye_outer", "right_eye_inner", "right_eye_upper", "right_eye_lower") if visible(pts, n)]
    mouth = [xy(pts, n) for n in ("mouth_left", "mouth_right", "lip_upper", "lip_lower") if visible(pts, n)]
    if not (eye_l and eye_r and mouth):
        return
    eye_c = mean(eye_l + eye_r)
    axis = sub(mean(mouth), eye_c)
    scale = norm(axis)
    n = lateral_unit(axis)
    if n is None or scale < 1:
        return
    a_unit = (axis[0] / scale, axis[1] / scale)
    thr = LR_THRESHOLD * scale
    p_eye = dot(sub(mean(eye_l), mean(eye_r)), n)
    if abs(p_eye) >= thr and p_eye < 0:
        rep.warn(
            f"{tag}: mắt trái (`left_eye_*`) nằm bên TRÁI ảnh. Với mặt nhìn về camera, mắt trái "
            "của người nằm bên PHẢI ảnh: có thể đổi trái/phải cả mặt (`lr_swap`)."
        )
    if visible(pts, "mouth_left") and visible(pts, "mouth_right"):
        p_m = dot(sub(xy(pts, "mouth_left"), xy(pts, "mouth_right")), n)
        if abs(p_m) >= thr and abs(p_eye) >= thr and (p_m > 0) != (p_eye > 0):
            rep.warn(f"{tag}: khóe miệng trái/phải ngược phía so với hai mắt (`lr_swap` một cặp?).")
    for side in ("left", "right"):
        outer, inner = f"{side}_eye_outer", f"{side}_eye_inner"
        if visible(pts, outer) and visible(pts, inner):
            d_out = abs(dot(sub(xy(pts, outer), eye_c), n))
            d_in = abs(dot(sub(xy(pts, inner), eye_c), n))
            if d_out < d_in:
                rep.warn(f"{tag}: `{outer}` gần sống mũi hơn `{inner}`: có thể đặt nhầm khóe trong / khóe ngoài.")
        upper, lower = f"{side}_eye_upper", f"{side}_eye_lower"
        if visible(pts, upper) and visible(pts, lower):
            if dot(sub(xy(pts, upper), xy(pts, lower)), a_unit) > 0:
                rep.warn(f"{tag}: `{upper}` nằm dưới `{lower}`: có thể đặt nhầm mí trên / mí dưới.")


def check_duplicates(items, rep, img_tag):
    """items: list (tag, pts, scale). Cảnh báo hai skeleton gần như trùng nhau."""
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ti, pi, si = items[i]
            tj, pj, sj = items[j]
            common = [n for n in pi if pi[n][2] > 0 and pj.get(n, (0, 0, 0))[2] > 0]
            if len(common) < 4:
                continue
            d = sum(norm(sub(xy(pi, n), xy(pj, n))) for n in common) / len(common)
            if d < 0.05 * max(si, sj, 1):
                rep.warn(f"{img_tag}: {ti} và {tj} gần như trùng nhau. Hai skeleton cho cùng một người? (`extra`)")


# ---------------------------------------------------------------- kiểm một file


def check_file(path: Path, kind: str, schema_all: dict, expected_images: int, rep: Report) -> dict:
    """kind = 'person' hoặc 'face'."""
    print(f"\n=== {path} (label `{kind}`) ===")
    docs = load_docs(path, rep)
    schema = schema_all[kind]
    stats = {
        "images": 0,
        "skeletons": 0,
        "per_image": {},
        "v": Counter(),
        "small": 0,
        "kp_len": 3 * len(schema["keypoints"]),
    }
    for doc_name, doc in docs:
        where = Path(doc_name).name
        cats = {c.get("id"): c for c in doc.get("categories", []) if isinstance(c, dict)}
        index_by_cat = {}
        for cid, cat in cats.items():
            if cat.get("name") in schema_all:
                index_by_cat[cid] = check_category(cat, schema_all, rep, where)
        label_ok = any(cats[c].get("name") == kind and idx is not None for c, idx in index_by_cat.items())
        if not label_ok:
            if not any(c.get("name") == kind for c in cats.values()):
                rep.error(f"{where}: không có label `{kind}` trong file. Xuất nhầm task hoặc nhầm định dạng?")
            print(f"  (Bỏ qua kiểm từng ảnh trong {where} vì label `{kind}` không hợp lệ.)")
            stats["images"] += len(doc.get("images", []))
            continue
        images = {im.get("id"): im for im in doc.get("images", []) if isinstance(im, dict)}
        by_image = defaultdict(list)
        for ann in doc.get("annotations", []):
            by_image[ann.get("image_id")].append(ann)
        stats["images"] += len(images)

        for img_id, img in sorted(images.items(), key=lambda kv: str(kv[1].get("file_name"))):
            fname = img.get("file_name", f"image_id={img_id}")
            anns = by_image.get(img_id, [])
            mine = []
            for k, ann in enumerate(anns, 1):
                cat = cats.get(ann.get("category_id"), {})
                cname = cat.get("name")
                tag = f"{fname} #{k}"
                if cname != kind:
                    if cname in schema_all:
                        rep.error(f"{tag}: skeleton label `{cname}` nằm trong file của task `{kind}`. Xóa nó trên CVAT.")
                    elif not ann.get("keypoints"):
                        rep.warn(f"{tag}: object không phải skeleton (label `{cname}`), bị bỏ qua.")
                    else:
                        rep.error(f"{tag}: label `{cname}` không thuộc bài lab.")
                    continue
                index = index_by_cat.get(ann.get("category_id"))
                if index is None:
                    continue
                pts = parse_points(ann, index, schema["keypoints"], rep, tag)
                if pts is None:
                    continue
                stats["skeletons"] += 1
                for p in pts.values():
                    stats["v"][p[2]] += 1
                nk = sum(1 for p in pts.values() if p[2] > 0)
                if nk == 0:
                    rep.warn(f"{tag}: mọi điểm đều outside (num_keypoints = 0). Skeleton này bị bỏ khi huấn luyện; nên xóa.")
                    continue
                check_bounds(pts, img, rep, tag)
                vis = [xy(pts, n) for n in pts if pts[n][2] > 0]
                ys = [p[1] for p in vis]
                xs = [p[0] for p in vis]
                height = max(ys) - min(ys)
                scale = max(height, max(xs) - min(xs))
                if kind == "person":
                    bbox = ann.get("bbox")
                    if isinstance(bbox, list) and len(bbox) == 4:
                        height = max(height, float(bbox[3]))
                    if height < MIN_PERSON_HEIGHT:
                        stats["small"] += 1
                        rep.warn(
                            f"{tag}: các điểm chỉ trải cao {height:.0f} px (< {MIN_PERSON_HEIGHT} px). "
                            "Người dưới 60 px không gán; nếu người cao hơn nhưng bị cắt khung thì bỏ qua cảnh báo này."
                        )
                    check_person_lr(pts, schema, rep, tag)
                else:
                    check_face(pts, rep, tag)
                mine.append((f"#{k}", pts, scale))
            stats["per_image"][fname] = len(mine)
            if kind == "person" and not mine:
                rep.warn(f"{fname}: không có skeleton `person` nào. Ảnh nào cũng có ít nhất một người cần gán.")
            if kind == "face" and len(mine) != 1:
                rep.warn(f"{fname}: có {len(mine)} skeleton `face`, mỗi ảnh mặt cần đúng 1.")
            check_duplicates(mine, rep, fname)

    if stats["images"] and stats["skeletons"] == 0:
        rep.error(f"File {path.name} không có skeleton `{kind}` nào. Bạn đã gán và bấm Ctrl+S trước khi export chưa?")
    if expected_images and stats["images"] != expected_images:
        rep.warn(f"File có {stats['images']} ảnh, bài giao {expected_images} ảnh. Xuất đúng task chưa?")
    return stats


def print_stats(kind: str, path: Path, st: dict) -> None:
    total = sum(st["v"].values()) or 1
    print(f"- File {kind}: {path.name}")
    print(f"  Số ảnh: {st['images']} · số skeleton `{kind}`: {st['skeletons']} · mỗi annotation {st['kp_len']} số")
    if st["per_image"]:
        counts = ", ".join(f"{Path(f).stem}={n}" for f, n in st["per_image"].items())
        print(f"  Skeleton theo ảnh: {counts}")
    print(
        "  Cờ v: "
        f"v=2 {st['v'][2]} ({100 * st['v'][2] / total:.0f}%) · "
        f"v=1 {st['v'][1]} ({100 * st['v'][1] / total:.0f}%) · "
        f"v=0 {st['v'][0]} ({100 * st['v'][0] / total:.0f}%)"
    )
    if st["skeletons"]:
        print(f"  num_keypoints trung bình: {(st['v'][1] + st['v'][2]) / st['skeletons']:.1f}")


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="Tự kiểm file xuất COCO Keypoints 1.0 của Lab Ngày 4.")
    ap.add_argument("--persons", type=Path, help="zip (hoặc .json) xuất từ task persons")
    ap.add_argument("--faces", type=Path, help="zip (hoặc .json) xuất từ task faces")
    ap.add_argument("--expected-person-images", type=int, default=12, help="số ảnh người được giao (mặc định 12; bonus B1: 3)")
    ap.add_argument("--expected-face-images", type=int, default=8, help="số ảnh mặt được giao (mặc định 8; bonus B2: 5)")
    args = ap.parse_args()
    if not args.persons and not args.faces:
        ap.error("cần ít nhất một trong --persons, --faces")

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    rep = Report()
    results = []
    if args.persons:
        results.append(("person", args.persons, check_file(args.persons, "person", schema, args.expected_person_images, rep)))
    if args.faces:
        results.append(("face", args.faces, check_file(args.faces, "face", schema, args.expected_face_images, rep)))

    print("\n----- TÓM TẮT (dán vào báo cáo mục 5) -----")
    for kind, path, st in results:
        print_stats(kind, path, st)
    if rep.v0_with_xy:
        print(
            f"- Ghi chú: {rep.v0_with_xy} điểm v = 0 vẫn mang tọa độ cũ (CVAT 2.x xuất như vậy; "
            "chuẩn COCO ghi 0, 0). Không phải lỗi: điểm v = 0 bị bỏ qua khi huấn luyện và khi chấm."
        )
    print(f"- Số [LỖI]: {len(rep.errors)} · số [CẢNH BÁO]: {len(rep.warnings)}")
    if rep.errors:
        print(f"Kết quả: CHƯA ĐẠT ({len(rep.errors)} lỗi). Sửa trên CVAT rồi xuất lại.")
        return 1
    print("Kết quả: ĐẠT" + (" (đọc lại từng cảnh báo ở trên)" if rep.warnings else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
