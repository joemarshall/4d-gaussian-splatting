from pathlib import Path

from utils.build_depth import get_min_depth_maps

output_path = Path("output/9moving")
get_min_depth_maps(output_path)