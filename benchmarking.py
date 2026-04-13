from PIL import Image
import numpy as np
import torch
import tempfile
import time
from pathlib import Path

def benchmark_cache_build_pil(args):
    from_path = Path(args[1])
    time_start = time.time()
    img_count =0 
    for x in from_path.glob("*.png"):
        with Image.open(x) as image_load:
            with tempfile.TemporaryDirectory() as temp_dir:
                out_path = Path(temp_dir) / f"{x.stem}.pt"
                im_data = np.array(image_load.convert("RGBA"))
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + np.array([1,1,1]) * (1 - norm_data[:, :, 3:4])
                image_load = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")
                image_tensor = torch.from_numpy(np.array(image_load)).permute(2, 0, 1).float() / 255.0
                torch.save(image_tensor, out_path)
                img_count += 1
                print("Cached image:", x.name, "to", out_path, "size:", image_tensor.shape)
    time_end = time.time()
    print(f"Converted {img_count} images in {time_end - time_start:.2f} seconds ({img_count / (time_end - time_start):.2f} images/second)")

def benchmark_cache_build_pywuffs(args):
    from pywuffs import ImageDecoderType, PixelFormat
    from pywuffs.aux import (
        ImageDecoder,
        ImageDecoderConfig,
        ImageDecoderFlags
    )

    config = ImageDecoderConfig()

    # All decoders are enabled by default
    config.enabled_decoders = [ImageDecoderType.PNG]
    config.pixel_format = PixelFormat.BGRA_PREMUL 
    decoder = ImageDecoder(config)



    from_path = Path(args[1])
    time_start = time.time()
    img_count =0 
    for x in from_path.glob("*.png"):
        decoding_result = decoder.decode(str(x))

        # Decoded image data in BGR format
        im_data = decoding_result.pixbuf

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / f"{x.stem}.pt"
            norm_data = im_data / 255.0
            image_tensor = torch.from_numpy(norm_data).permute(2, 0, 1).float()[:3, ...]
            torch.save(image_tensor, out_path)
            img_count += 1
            print("Cached image:", x.name, "to", out_path, "size:", image_tensor.shape)
    time_end = time.time()
    print(f"Converted {img_count} images in {time_end - time_start:.2f} seconds ({img_count / (time_end - time_start):.2f} images/second)")


if __name__ == "__main__":
    import sys
    benchmark_cache_build_pywuffs(sys.argv)


