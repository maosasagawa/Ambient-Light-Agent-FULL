# image_processor.py

from typing import Any

from PIL import Image


def process_image_to_led_data(
    image: Image.Image, resolution: tuple[int, int] = (16, 16)
) -> dict[str, Any]:
    """
    将Pillow图像处理成LED灯珠数据，同时生成raw和json两种格式。

    :param image: 输入的Pillow Image对象。
    :param resolution: 目标LED分辨率, e.g., (16, 16)。
    :return: 包含 'raw' (二进制) 和 'json' 两种格式数据的字典。
    """
    # 1. 高质量下采样到目标分辨率
    led_colors_image = image.resize(resolution, Image.Resampling.LANCZOS)
    
    width, height = led_colors_image.size
    
    # 2. 准备数据容器
    raw_data = bytearray(width * height * 3)
    json_pixel_data = []
    
    # 3. 遍历像素并填充数据
    for y in range(height):
        row = []
        for x in range(width):
            r, g, b = led_colors_image.getpixel((x, y))
            
            # 为STM32准备的原始二进制数据 (R, G, B, R, G, B, ...)
            # 这种格式对于内存有限的MCU来说效率最高
            idx = (y * width + x) * 3
            raw_data[idx] = r
            raw_data[idx + 1] = g
            raw_data[idx + 2] = b
            
            # 为Web前端准备的JSON数据 [[r,g,b], [r,g,b], ...]
            row.append([r, g, b])
        json_pixel_data.append(row)

    return {
        "raw": raw_data,
        "json": {
            "width": width,
            "height": height,
            "pixels": json_pixel_data # 嵌套列表，前端易于渲染
        }
    }