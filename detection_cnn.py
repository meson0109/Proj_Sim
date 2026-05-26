"""
Pascal VOC 2007 目标检测实验辅助函数，检测模型使用 CNN 系列的 YOLO。

整体调用逻辑：
1. notebook 调用 train_yolo_voc(...)
2. train_yolo_voc 先调用 _require_ultralytics() 检查 YOLO 依赖
3. train_yolo_voc 再调用 resolve_voc_eval_split(...) 决定用 test 还是 val 评估
4. train_yolo_voc 调用 convert_voc2007_to_yolo(...) 把 VOC XML 数据转成 YOLO 数据格式
5. convert_voc2007_to_yolo 对每张图片调用 parse_voc_annotation(...) 读取 XML 标注
6. parse_voc_annotation 内部读出 VOC 框 xmin/ymin/xmax/ymax
7. convert_voc2007_to_yolo 调用 voc_box_to_yolo(...) 把 VOC 框转换成 YOLO 的归一化中心点格式
8. train_yolo_voc 用 ultralytics.YOLO(model_name) 训练模型，并调用 model.val(...) 得到 mAP
9. notebook 可调用 summarize_map(metrics) 把 mAP、precision、recall 整理成字典
10. 训练完成后，可调用 predict_test_folder(...) 对 VOC2007test/JPEGImages 或自定义图片文件夹画预测框

重要说明：
- 如果只下载了 VOCtrainval_06-Nov-2007.tar，通常没有 test.txt，本文件会自动改用 val.txt 评估。
- 如果又下载并解压了 VOCtest_06-Nov-2007.tar，就可以使用官方 test split。
- 如果你把 trainval 和 test 解压成两个目录，可以在 train_yolo_voc 中传 eval_voc_root。
"""

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def _require_ultralytics():
    """
    检查当前 Python 环境里是否安装了 ultralytics。

    ultralytics 提供 YOLO 类，后面的训练、验证、预测都依赖它。
    如果没有安装，这里会抛出更清晰的错误，提醒在当前 notebook 环境中执行：
    pip install ultralytics
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for the YOLO experiment. Install it in the "
            "notebook environment with: pip install ultralytics"
        ) from exc
    return YOLO


def voc_box_to_yolo(box, width, height):
    """
    把 Pascal VOC 的边界框格式转换成 YOLO 标签格式。

    VOC XML 中的框是绝对像素坐标：
    xmin, ymin, xmax, ymax

    YOLO 标签文件要求每一行是：
    class_id x_center y_center box_width box_height

    其中 x_center、y_center、box_width、box_height 都要除以图片宽高，
    变成 0 到 1 之间的归一化数值。
    """
    xmin, ymin, xmax, ymax = box
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_w = (xmax - xmin) / width
    box_h = (ymax - ymin) / height
    return x_center, y_center, box_w, box_h


def parse_voc_annotation(xml_path):
    """
    读取一张 VOC 图片对应的 XML 标注文件。

    输入：
    - xml_path: 例如 VOC2007/Annotations/000007.xml

    做的事情：
    - 读取图片宽高
    - 遍历 XML 中所有 object
    - 取出类别名、difficult 标记、边界框坐标
    - 把类别名映射成 0-19 的类别编号

    返回：
    - width: 图片宽度
    - height: 图片高度
    - objects: 列表，每个元素为 (cls_id, difficult, (xmin, ymin, xmax, ymax))
    """
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    objects = []
    for obj in root.findall("object"):
        cls = obj.findtext("name")
        if cls not in VOC_CLASSES:
            continue
        difficult = int(obj.findtext("difficult", default="0"))
        box = obj.find("bndbox")
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))
        objects.append((VOC_CLASSES.index(cls), difficult, (xmin, ymin, xmax, ymax)))
    return width, height, objects


def convert_voc2007_to_yolo(
    voc_root,
    out_dir="outputs/voc2007_yolo",
    splits=("trainval", "test"),
    split_roots=None,
):
    """
    把 VOC 原始数据转成 Ultralytics YOLO 可直接读取的数据格式。

    输入：
    - voc_root: VOC2007 根目录，例如 proj_sim/VOCdevkit/VOC2007
    - out_dir: 转换后的 YOLO 数据输出目录
    - splits: 要转换的数据划分，例如 ("trainval", "val") 或 ("trainval", "test")
    - split_roots: 可选字典，用于训练集和测试集不在同一目录的情况。
      例如 {"trainval": VOC2007trainval路径, "test": VOC2007test路径}

    VOC 原始结构：
    从：
    proj_sim/VOCdevkit/VOC2007/
    ├── JPEGImages
    ├── Annotations
    └── ImageSets/Main

    YOLO 输出结构：
    转换到：
    proj_sim/outputs/voc2007_yolo/
    ├── images/trainval
    ├── images/val
    ├── labels/trainval
    ├── labels/val
    └── voc2007.yaml

    关键处理：
    - images/... 下面复制原 jpg 图片
    - labels/... 下面生成同名 txt 标签
    - 每个 txt 标签的一行表示一个目标：
      class_id x_center y_center width height
    - 最后生成 voc2007.yaml，告诉 YOLO 训练集、验证集和类别名在哪里

    返回：
    - yaml_path: 生成的 voc2007.yaml 路径
    """
    voc_root = Path(voc_root)
    out_dir = Path(out_dir)
    split_roots = split_roots or {}
    default_ann_dir = voc_root / "Annotations"
    if not default_ann_dir.exists():
        raise FileNotFoundError(f"Cannot find VOC annotations at {default_ann_dir}")

    for split in splits:
        current_root = Path(split_roots.get(split, voc_root))
        ann_dir = current_root / "Annotations"
        img_dir = current_root / "JPEGImages"
        split_dir = current_root / "ImageSets" / "Main"
        # split 文件给出这个划分包含哪些图片 id，例如 trainval.txt、val.txt。
        split_file = split_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Cannot find VOC split file: {split_file}. "
                "If you only downloaded VOCtrainval_06-Nov-2007.tar, use split='val' "
                "for validation, or download VOCtest_06-Nov-2007.tar for split='test'."
            )
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        for image_id in ids:
            # 复制图片到 YOLO 的 images/split 目录。
            src_img = img_dir / f"{image_id}.jpg"
            dst_img = out_dir / "images" / split / src_img.name
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            # 读取 XML 标注，转换每个框，写成 YOLO txt 标签。
            width, height, objects = parse_voc_annotation(ann_dir / f"{image_id}.xml")
            label_lines = []
            for cls_id, difficult, box in objects:
                # difficult=1 的样本通常不参与训练，避免模糊标注影响模型。
                if difficult:
                    continue
                xc, yc, bw, bh = voc_box_to_yolo(box, width, height)
                label_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            (out_dir / "labels" / split / f"{image_id}.txt").write_text("\n".join(label_lines))

    eval_split = splits[1] if len(splits) > 1 else splits[0]
    yaml_path = out_dir / "voc2007.yaml"
    # Ultralytics 通过这个 yaml 找数据集路径和类别名。
    # 这里把 test 也指向 eval_split，是为了 model.val(split="test") 时能正常评估。
    yaml_path.write_text(
        "path: " + str(out_dir.resolve()) + "\n"
        "train: images/trainval\n"
        f"val: images/{eval_split}\n"
        f"test: images/{eval_split}\n"
        "names:\n"
        + "\n".join(f"  {i}: {name}" for i, name in enumerate(VOC_CLASSES))
        + "\n"
    )
    return yaml_path


def resolve_voc_eval_split(voc_root, preferred="test"):
    """
    决定目标检测实验用哪个划分做评估。

    默认希望使用 test.txt，这对应官方 VOC test split。
    但如果只下载了 VOCtrainval_06-Nov-2007.tar，目录里通常没有 test.txt，
    只有 train.txt、val.txt、trainval.txt。

    所以这里的策略是：
    - 如果 preferred 对应的文件存在，例如 test.txt，就用它
    - 否则如果 val.txt 存在，就自动回退到 val
    - 两者都没有，说明 VOC 解压路径或数据集不完整
    """
    split_dir = Path(voc_root) / "ImageSets" / "Main"
    preferred_file = split_dir / f"{preferred}.txt"
    if preferred_file.exists():
        return preferred
    val_file = split_dir / "val.txt"
    if val_file.exists():
        print(
            f"VOC split '{preferred}' not found at {preferred_file}; "
            "using 'val' split instead."
        )
        return "val"
    raise FileNotFoundError(
        f"Cannot find {preferred_file} or {val_file}. "
        "Check that VOCtrainval_06-Nov-2007.tar was extracted correctly."
    )


def train_yolo_voc(
    voc_root,
    out_dir="outputs/voc2007_yolo",
    model_name="yolov8n.pt",
    epochs=30,
    imgsz=640,
    batch=8,
    device=None,
    eval_split="test",
    eval_voc_root=None,
):
    """
    训练 YOLO 目标检测网络，并返回训练对象和评估指标。

    输入：
    - voc_root: VOC2007 根目录
    - out_dir: 转换后数据、训练日志、权重的输出目录
    - model_name: YOLO 初始模型，例如 yolov8n.pt
    - epochs: 训练轮数
    - imgsz: 输入图片尺寸，YOLO 会把图片缩放到该尺寸训练
    - batch: batch size
    - device: 训练设备，GPU 通常传 0，CPU 传 "cpu"
    - eval_split: 优先评估划分，默认 test；没有 test 时会自动用 val
    - eval_voc_root: 可选。若 VOCtest_06-Nov-2007.tar 单独解压到了另一个目录，
      这里传测试集 VOC2007 根目录；不传则默认训练和评估都从 voc_root 里找。

    做的事情：
    1. 检查 ultralytics 是否安装
    2. 决定评估 split
    3. 把 VOC 转换成 YOLO 数据格式
    4. 加载预训练 YOLO 模型
    5. 训练模型
    6. 在评估 split 上计算 precision、recall、mAP 等指标

    返回：
    - model: 训练后的 YOLO 模型对象
    - train_result: ultralytics 的训练结果对象
    - metrics: ultralytics 的验证指标对象，可交给 summarize_map 整理
    """
    YOLO = _require_ultralytics()
    eval_root = Path(eval_voc_root) if eval_voc_root is not None else Path(voc_root)
    eval_split = resolve_voc_eval_split(eval_root, preferred=eval_split)
    split_roots = {"trainval": Path(voc_root), eval_split: eval_root}
    yaml_path = convert_voc2007_to_yolo(
        voc_root,
        out_dir=out_dir,
        splits=("trainval", eval_split),
        split_roots=split_roots,
    )
    model = YOLO(model_name)
    train_result = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(Path(out_dir) / "runs"),
        name="yolo_voc2007",
    )
    metrics = model.val(data=str(yaml_path), split="test", imgsz=imgsz, device=device)
    return model, train_result, metrics


def predict_test_folder(
    weights,
    image_folder="./PascalVOC-Test",
    out_dir="outputs/pascalvoc_test_predictions",
    conf=0.25,
    imgsz=640,
    device=None,
    max_images=None,
    save_images=False,
    return_images=False,
):
    """
    对指定文件夹中的图片做目标检测，并保存画框结果。

    输入：
    - weights: 训练好的权重路径，例如 runs/yolo_voc2007/weights/best.pt
    - image_folder: 待检测图片文件夹，例如 VOC2007test/JPEGImages 或 ./PascalVOC-Test
    - out_dir: 保存可视化检测结果的目录
    - conf: 置信度阈值，低于这个值的预测框会被过滤
    - imgsz: 推理时的图片尺寸
    - device: 推理设备
    - max_images: 最多检测多少张图片；用于 2.3 展示时通常设为 3
    - save_images: 是否把画好框的结果图保存到 out_dir/images
    - return_images: 是否把画好框的图像数组随结果一起返回，方便 notebook 直接显示

    输出：
    - predictions: 列表，每个元素是 (图片文件名, 检测结果列表)
      检测结果里包含 class、score、xyxy 像素坐标
    - 如果 return_images=True，则返回 (predictions, rendered_images)
      rendered_images 中每个元素是 (图片文件名, RGB图像数组)

    当 save_images=True 时，ultralytics 会把画好预测框的图片保存到：
    out_dir/images/
    """
    YOLO = _require_ultralytics()
    model = YOLO(str(weights))
    image_folder = Path(image_folder)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        p for p in image_folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if max_images is not None:
        image_paths = image_paths[:max_images]
    predictions = []
    rendered_images = []
    for result in model.predict(
        source=[str(p) for p in image_paths],
        conf=conf,
        imgsz=imgsz,
        device=device,
        save=save_images,
        project=str(out_dir),
        name="images",
        exist_ok=True,
    ):
        labels = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            score = float(box.conf.item())
            xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            labels.append({"class": model.names[cls_id], "score": score, "xyxy": xyxy})
        predictions.append((Path(result.path).name, labels))
        if return_images:
            # Ultralytics 的 plot() 返回已画框图像，可直接在 notebook 中 imshow。
            rendered_images.append((Path(result.path).name, result.plot()))
    if return_images:
        return predictions, rendered_images
    return predictions


def summarize_map(metrics):
    """
    从 ultralytics 的 metrics 对象里提取报告常用指标。

    返回字段：
    - mAP50: IoU 阈值为 0.5 时的 mean Average Precision
    - mAP50-95: IoU 从 0.5 到 0.95 多个阈值平均的 mAP，更严格
    - precision: 预测为目标的框里有多少是真的
    - recall: 真实目标中有多少被检测出来
    """
    box = metrics.box
    return {
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }
