import sys
from pathlib import Path
import torch.onnx
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

    lang_char = get_lang_char(["en"], model)
    print(lang_char)

    sys.exit(0)

    # model input
    x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)

    # model output
    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(
        torch_model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        "super_resolution.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )
