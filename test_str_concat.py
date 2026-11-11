import timeit
from unittest import result
import numpy as np
import os
import struct
import itertools

def current_version(data):
    result = ""
    for i,c in enumerate(data):
        result += f"{int(c)},"
        if i % 16 == 15:
            result += "\t\n"
    return result

def list_of_strings(data):
    lines = []
    cur_line = ""
    for i, c in enumerate(data):
        cur_line += f"{int(c)},"
        if i % 16 == 15:
            lines.append(cur_line) 
            cur_line = ""
    return "\t\n".join(lines)

def line_at_a_time(data):
    result = []
    for i in range(0,len(data)-16,16):
        result.append(f"{data[i]%0xff},{(data[i+1])%0xff},{(data[i+2])%0xff},{(data[i+3])%0xff},{(data[i+4])%0xff},{(data[i+5])%0xff},{(data[i+6])%0xff},{(data[i+7])%0xff},{(data[i+8])%0xff},{(data[i+9])%0xff},{(data[i+10])%0xff},{(data[i+11])%0xff},{(data[i+12])%0xff},{(data[i+13])%0xff},{(data[i+14])%0xff},{(data[i+15])%0xff},")
    return "\t\n".join(result)


def struct_unpack(data):
    result = []
    i=0
    while i < len(data):
        if i+16<len(data):
            line = struct.unpack('16B',data[i:i+16])
            i+=16
        else:
            line = struct.unpack(f'{len(data)-i}B',data[i:i+len(data)-i])
            i=len(data)
        result.append(str(line).replace(" ", "").strip("()")+",\t\n")
    return "".join(result)

def use_map(data):
    result = []
    i=0
    for i in range(0,len(data),16):
        end_pos = min(i+16,len(data)) 
        line_data = data[i:end_pos]
        line = map(lambda x: str(x),line_data)
        result.append(",".join(line)+",\t\n")

    return "".join(result)

def list_comp(data):
    result = []
    i=0
    for i in range(0,len(data),16):
        end_pos = min(i+16,len(data)) 
        line_data = data[i:end_pos]
        line = [str(x) for x in line_data]
        result.append(",".join(line))

    return ",\n".join(result)

def itertools_map(data):
    return ",\n".join(str(batch).strip("()") for batch in itertools.batched(data,16))


def time_version(name,data):
    time = timeit.timeit(f'{name}({data})', number=1, globals=globals())
    print(f"{name}({len(data)}) took {time:.6f} seconds")

def test_version(name,data):
    result = globals()[name](data)
    print(f"{name}({len(data)}) produced {len(result)} characters",flush=True)
    print(result)

all_versions = [ "itertools_map","list_comp","use_map","list_of_strings", "line_at_a_time",  "struct_unpack"]#,"current_version"]

data = os.urandom(512)
for v in all_versions:
    test_version(v, data)


# In my model I'm testing number of consts = 104691584
for n in [1, 10, 100, 1000, 10000,100000,1000000,10000000,100000000]:
    data = os.urandom(n)
    print("Testing with n =", n)
    for v in all_versions:
        time_version(v, data)
