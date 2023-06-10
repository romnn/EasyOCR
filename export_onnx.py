import onnx
import onnxruntime
import sys
import cv2
from pathlib import Path
from pprint import pprint
import numpy as np
import torch.onnx
import easyocr.utils
import easyocr.imgproc
import easyocr.detection as craft_detector
from easyocr.config import MODULE_PATH, BASE_PATH, detection_models, recognition_models


def get_lang_char(lang_list, model):
    lang_char = []
    for lang in lang_list:
        char_file = Path(BASE_PATH) / "character" / (lang + "_char.txt")
        with open(char_file, "r", encoding="utf-8-sig") as input_file:
            char_list = input_file.read().splitlines()
        lang_char += char_list
    if model.get("symbols"):
        symbol = model["symbols"]
    elif model.get("character_list"):
        symbol = model["character_list"]
    else:
        symbol = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
    lang_char = set(lang_char).union(set(symbol))
    lang_char = "".join(lang_char)
    return lang_char


def get_detector_path(model_dir, detector_name):
    detector_path = model_dir / detection_models[detector_name]["filename"]
    assert detector_path.is_file()
    return detector_path


def get_num_params(model, learnable=False):
    pp = 0
    for p in list(model.parameters()):
        if learnable and not p.requires_grad:
            continue
        nn = 1
        for s in list(p.size()):
            nn *= s
        pp += nn
    return pp


def overlay_boxes_on_image(img, boxes):
    for bbox in boxes:
        bbox = np.array(bbox, dtype=np.int32)
        print(bbox)
        img = cv2.polylines(
            img,
            [bbox],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )
    return img


if __name__ == "__main__":
    print(MODULE_PATH)

    quantize = True

    model_storage_directory = Path(MODULE_PATH) / "model"
    model_storage_directory.mkdir(parents=True, exist_ok=True)

    user_network_directory = Path(MODULE_PATH) / "user_network"
    user_network_directory.mkdir(parents=True, exist_ok=True)
    # sys.path.append(user_network_directory)

    model_lang = "english"
    model = recognition_models["gen2"]["english_g2"]
    recog_network = "generation2"
    character = model["characters"]

    model_path = model_storage_directory / model["filename"]
    assert model_path.is_file()

    lang_list = ["en"]
    lang_char = get_lang_char(lang_list, model)
    print(lang_char)

    dict_list = dict()
    for lang in lang_list:
        dict_list[lang] = Path(BASE_PATH) / "dict" / (lang + ".txt")

    detector_path = get_detector_path(model_storage_directory, "craft")

    device = "cuda"
    detector = craft_detector.get_detector(
        detector_path,
        device=device,
        quantize=quantize,
        cudnn_benchmark=False,
    )
    print(
        "detector has %d params (%d learnable)"
        % (
            get_num_params(detector, learnable=False),
            get_num_params(detector, learnable=True),
        )
    )

    input_image = Path(BASE_PATH) / "../examples/english.png"
    print(input_image)
    assert input_image.is_file()
    # image, img_cv_grey = easyocr.utils.reformat_input(str(input_image))
    img_cv_grey = cv2.imread(str(input_image), cv2.IMREAD_GRAYSCALE)
    image = easyocr.imgproc.loadImage(str(input_image))

    print(image.shape)

    if isinstance(image, np.ndarray) and len(image.shape) == 4:  # image is batch of np arrays
        image_arrs = image
    else:  # image is single numpy array
        image_arrs = [image]

    img_resized_list = []
    # resize
    for img in image_arrs:
        img_resized, target_ratio, size_heatmap = easyocr.imgproc.resize_aspect_ratio(
            img, square_size=2560, interpolation=cv2.INTER_LINEAR, mag_ratio=1.0
        )
        img_resized_list.append(img_resized)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = [np.transpose(easyocr.imgproc.normalizeMeanVariance(n_img), (2, 0, 1)) for n_img in img_resized_list]
    x = torch.from_numpy(np.array(x))
    x = x.to(device)

    # forward pass
    with torch.no_grad():
        y, feature = detector(x)

    print(y.shape, feature.shape)
    # sys.exit(0)

    if False:
        recog_network_params = {
            "input_channel": 1,
            "output_channel": 256,
            "hidden_size": 256,
        }

        import easyocr.recognition

        separator_list = {}
        recognizer, converter = easyocr.recognition.get_recognizer(
            recog_network,
            recog_network_params,
            character,
            separator_list,
            dict_list,
            model_path,
            device=device,
            quantize=quantize,
        )
        print(recognizer)

    # model input
    # x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)

    # model output
    # torch_out = detector(x)
    print(x.shape)

    # Export the model
    onnx_model_path = "craft_detector.onnx"
    torch.onnx.export(
        detector.module,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        onnx_model_path,
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(y), ort_outs[0], rtol=1e-03, atol=1e-05)

    (boxes_list, polys_list) = craft_detector.compute_boxes_and_polys(
        torch.from_numpy(ort_outs[0]).to(device),
        # ort_outs[0].cuda(),
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        poly=False,
        estimate_num_chars=False,
        ratio_w=ratio_w,
        ratio_h=ratio_h,
    )
    # print(len(boxes_list))
    boxes_list = boxes_list[0]
    # print([b.shape for b in boxes_list])
    # pprint(boxes_list.unsqueeze())
    ocr_image = overlay_boxes_on_image(image.copy(), boxes_list)
    cv2.imwrite("./onnx_python_overlay.jpg", ocr_image)
    # pprint(boxes_list)
    # pprint(polys_list)

    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
