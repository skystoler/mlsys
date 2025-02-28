import time
import torch
import torch.nn.functional as F
from ..utils import image_utils

SOURCE_DIR = "/root/autodl-tmp/data/"
TRT_PATH = SOURCE_DIR + 'prt_ft13_emptyrecog_invert_0411_147_opt.ts'
PATH = SOURCE_DIR + 'uniocr_wfeat_240919134829_uniocr_ft47xkat6_240903_75.pth'
IMG_URL = SOURCE_DIR + 'text_rectification.jpg'

def prepare_input(img_url, batch=1, height = 448, width = 448, pad_value = 255, dtype = torch.float16, interpolation = "bicubic", is_vertical=True):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    padding_left_, padding_top_, padding_right_, padding_bottom_ = 0, 0, 0, 0
    min_resize_ratio_inv = 0.3
    output_height = height
    output_width = width
    if is_vertical:
        resize_ratio_ = max(float(width) / output_width, min_resize_ratio_inv)
        output_height_ = min(int(height / resize_ratio_), output_height)
        # 竖排文字底部填充
        padding_bottom_ = max(0, output_height - output_height_)
        # 竖排文字可以左右填充
        output_width_ = min(int(width / resize_ratio_), output_width)
        padding_left_ = max(0, (output_width - output_width_) // 2)
        padding_right_ = max(0, output_width - output_width_ - padding_left_)

    else:
        resize_ratio_ = max(float(height) / output_height, min_resize_ratio_inv)
        output_width_ = min(int(width / resize_ratio_), output_width)
        # 横排文字右侧填充
        padding_right_ = max(0, output_width - output_width_)
        # 横排文字可以上下填充
        output_height_ = min(int(height / resize_ratio_), output_height)
        padding_top_ = max(0, (output_height - output_height_) // 2)
        padding_bottom_ = max(0, output_height - output_height_ - padding_top_)

    image = image_utils.load_image(img_url)
    img_arr = image_utils.pil_to_numpy(image)
    image_t = image_utils.numpy_to_pt(img_arr)
    image_t = F.interpolate(image_t, (output_height_, output_width_), mode=interpolation, align_corners=True)
    if interpolation == 'bicubic':
        image_t = image_t.clamp(0, 255)
    image_t = F.pad(
        image_t, (padding_left_, padding_right_, padding_top_, padding_bottom_), value=pad_value
    )
    image_t.to(device)
    image_t.to(dtype)
    mask_t = torch.ones((batch, 1, output_height_, output_width_), device=device, dtype=dtype)
    token_ids_t = torch.ones((image_t.shape[0], 1), device=device, dtype=torch.int64)
    return image_t, mask_t, token_ids_t


class SageAttention():
    def __init__(self) -> None:
        pass

def replace_attn(module, att_mask):
    module_output = module
    if isinstance(module, torch.nn.modules.MultiheadAttention):
        # qkv = module.qkv
        # dim = qkv.weight.shape[1] * module.num_heads
        
        module_output = SageAttention(dim, module.num_heads, attn_mask=att_mask)
    for name, child in module.named_children():
        module_output.add_module(name, replace_attn(child, att_mask))
    del module
    return module_output


def main():
    model = torch.jit.load(TRT_PATH, map_location=torch.device("cuda:0"))
    model.eval()
    
    images_t, masks_t, token_ids_t = prepare_input(img_url=IMG_URL)
    
    """
    not compile
    """
    begin = time.time()
    output = model(images_t, masks_t, token_ids_t)
    end = time.time()
    
    print(f'Inference time: {end - begin:.3f}s')
    peak_mem = torch.cuda.max_memory_allocated()
    print(f'Peak memory: {peak_mem / 1024**3:.3f}GiB')
    
    """
    compile
    """
    replace_attn(model)
    # replace_attn(model.base_architecture. )
    begin = time.time()
    optimize_output = model(images_t, masks_t, token_ids_t)
    end = time.time()
    
    print(f'Inference time: {end - begin:.3f}s')
    peak_mem = torch.cuda.max_memory_allocated()
    print(f'Peak memory: {peak_mem / 1024**3:.3f}GiB')

    torch.testing.assert_close(output, optimize_output)
    
if __name__ == '__main__':
    main()