import time
import random
from pathlib import Path

from tqdm import tqdm


import torch
torch.set_float32_matmul_precision('high')

from utils.sh_utils import eval_shfs_4d,eval_sh

from utils.general_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    build_rotation_4d,
    build_scaling_rotation_4d,
)


# initialize from depth, ?clip by cameras?
# work out how to play it
# work out how to make it small
#
#
# total size of model:
# for each gaussian:
# x,y,z,t = 4
# scaling x,y,z,t = 4
# rotation 1: x,y,z,w = 4
# rotation 2: x,y,z,w = 4
# features_dc: 3
# features_rest: 47*3 = 141
# opacity: 1
# 
# total = 4+4+4+4+3+141+1 = 157 floats per gaussian
# in training, for each parameter we need:
# - parameter itself
# -  exp_avg
# -  exp_avg_sq
# i.e. 3* all above
#
# and densifiers:
# 1 floats for xyz gradient accum
# 1 floats for denom
# 1 floats for max_radii2D
# 1 floats for t gradient accum 
#
# so for each gaussian, we need 3 * 157 = 471 floats in training
# 
#
#
#  
#
#
# store on disk as fp16 / bf16 = 50% less data
# or even as 8 bit with custom compression per chunk
# todo: check conversion time
# todo: save sorted by mean of time
# todo: split out stationary gaussians in background vs foreground
# maybe: anything outside camera bounding box = stationary, zero time based parameters, gradient + fix rotation 
#  
#
# to train with big file
# need:
# gaussians.set_active_time_range()
# keep in memory:
# 1) gaussian time
# 2) gaussian covariances
# loads:
# 1) gaussians
# 2) optimizer state
# 3) state of densifiers if needed 
# -  n.b. in densifiers, have a state dirty flag, and don't worry about saving
# state if it is zero still (i.e. just post densification)
# 

#
# ignore gradients (assume it is just after an optimizer step)
#
# todo2: try different more dense initializers (e.g. random selection of frame, x, y, depth to make points?)
#       -n.b. can make sure initialisation happens spread out over time 
# 

# benchmark load/save <-> cuda on set_active_time_range
# and use that to work out how many training steps per time range make sense
# between switches

# TODO: run on the CVL cluster 


# can we stop training stationary points once we have found them? And does that help much performance-wise given we
# can't train the other points without rendering them. 
# or maybe we can if we depth-clip everything???








from utils.big_tensors import *






TIMES = []

def add_time(label,count = None):
    global TIMES
    last_time = TIMES[-1][2] if len(TIMES)>0 else time.monotonic()
    cur_time = time.monotonic()
    duration = cur_time - last_time
    if count is None:
        TIMES.append((label, duration, cur_time,None))
    else:
        TIMES.append((label, duration, cur_time,duration/count))

output_folder = Path("output/9moving/model_output/")
checkpoints = Path(output_folder).glob("*.pth")
sorted_checkpoints = list(sorted(checkpoints, key=lambda x: x.stat().st_mtime))
if len(sorted_checkpoints):
    latest_pth = sorted_checkpoints[-1]


add_time("start")
all_tensors = torch.load(latest_pth, map_location="cpu",mmap=True,weights_only = False)
add_time("loaded raw data")
chunked_tensor = TimeAndScaleChunkedTensor(device="cuda",min_chunk_size=.1)
tensor_set = {}
for k in all_tensors[0].keys():
    if type(all_tensors[0][k]) == torch.Tensor or type(all_tensors[0][k]) == torch.nn.Parameter:
        chunked_tensor.add_tracked_tensor(k)
        tensor_set[k] = all_tensors[0][k]

print(tensor_set['_t'])

def get_scale_from_tensors(tensor_set):
    scaling_xyzt = torch.exp(torch.cat([tensor_set['_scaling'], tensor_set['_scaling_t']], dim=1))
    L = build_scaling_rotation_4d(
        scaling_xyzt,
        tensor_set['_rotation'],
        tensor_set['_rotation_r']
    )
    actual_covariance = L @ L.transpose(1, 2)
    cov_t =actual_covariance[:, 3, 3]#.unsqueeze(1)
    sd_t = torch.sqrt(cov_t)
    # opacity multiplier is 0.05 at this point
    visible_range = 1.96 * sd_t
    # opacity multiplier is 0.01 at this point
    # visible_range = 2.576 * sd_t

    return visible_range

scales = get_scale_from_tensors(tensor_set)
print("Scales:",scales.shape)

chunked_tensor.update_chunks_from_device(tensor_set,tensor_set['_t'].squeeze(),scales)
print("Made initial chunks")
add_time("Made chunks")
for x in tqdm(range(100)):
    import random
    time_start=random.random()*5.5
    device_data,device_times,device_scales = chunked_tensor.move_chunks_to_device(min_time=time_start,max_time=time_start+0.5)
    if device_data is not None:
        chunked_tensor.update_chunks_from_device(device_data,device_times,device_scales)
add_time("Retrieved chunks",100)


# print(all_tensors[0].keys())
# time_vals = all_tensors[0]["_t"]
# print(torch.cuda.memory.memory_summary())
# add_time("Start selection")
# NUM_SELECTIONS = 1000
# for x in range(NUM_SELECTIONS):
#     start_time = random.uniform(0,4.5)
#     end_time = start_time+0.5
#     time_selection = (time_vals > start_time) & (time_vals < end_time)
#     time_selection = time_selection.squeeze()
#     selected_vars = []
#     for t in all_tensors[0].values():
#         #print(type(t))
#         if (type(t) == torch.Tensor or type(t) == torch.nn.Parameter) and t.shape[0] == time_vals.shape[0]:
#             selected = t[time_selection].detach().to(device="cuda")
#             selected_vars.append(selected)
#     #print(torch.cuda.memory.memory_allocated(),selected_vars)
# selected_vars = []
        

# add_time("Done selection",NUM_SELECTIONS)
# print(torch.cuda.memory.memory_summary())


for label,duration,_the_time,count in TIMES:
    if count is None:
        print(label,":",duration)
    else:
        print(label,":",duration,"time/it:",count)



import sys
sys.exit(0)

def dump_tensors(tensor_list, key_names = None, indent=0):
    total_floats=0
    floats_per_point = 0
    if key_names is None:
        key_names = [str(x) for x in range(len(tensor_list))]
    else:
        key_names = [str(x) for x in key_names]
    for x, key_name in zip(tensor_list, key_names):
        if type(x) == torch.Tensor or type(x) == torch.nn.Parameter:
            print(" "*indent, key_name,":", x.shape, x.dtype, x.device)
            if len(x.shape)>0:
                total_floats += x.numel()
                floats_per_point += x[0].numel() if x.numel()>0 else 0
        elif type(x) == list:
            print(" "*indent, key_name, ":","list of length", len(x))
            tf,fpp = dump_tensors(x,  indent=indent+2)
            total_floats+=tf
            floats_per_point+=fpp
        elif type(x) == tuple:
            print(" "*indent, key_name, ":","tuple of length", len(x))
            tf,fpp = dump_tensors(x,  indent=indent+2)
            total_floats+=tf
            floats_per_point+=fpp
        elif type(x) == dict:
            print(" "*indent, key_name, ":","dict with keys", list(x.keys()))
            tf,fpp = dump_tensors(list(x.values()),list(x.keys()),indent= indent+2)
            total_floats+=tf
            floats_per_point+=fpp
        else:
            print(" "*indent, key_name,":",str(x), type(x))
    return total_floats,floats_per_point

total_floats, floats_per_point = dump_tensors(all_tensors)

print("Total floats:", total_floats)
print("Floats per point:", floats_per_point)

# analysis of SH harmonics
# - plot means of SH coefficeints across all points, for each degree
sh_features= all_tensors[0]["_features_rest"]
sh_features = torch.cat([all_tensors[0]["_features_dc"], sh_features], dim=1)
sh_means = sh_features.mean(dim=0)
sh_max = sh_features.max(dim=0).values
sh_min = sh_features.min(dim=0).values
sh_variances = sh_features.var(dim=0)
print(f"SH Coefficient Means (shape: {sh_features.shape}):")
print(sh_means)
print(f"SH Coefficient Variances (shape: {sh_features.shape}):")
print(sh_variances)
print(f"SH Coefficient Max (shape: {sh_features.shape}):")
print(sh_max)
print(f"SH Coefficient Min (shape: {sh_features.shape}):")
print(sh_min)





# Correlation between RGB channels for each SH coefficient index (across all points)
# sh_features shape expected: [num_points, 47, 3]
r = sh_features[:, :, 0]
g = sh_features[:, :, 1]
b = sh_features[:, :, 2]

def per_value_corr(x, y, eps=1e-12):
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    num = (x * y).sum(dim=0)
    den = torch.sqrt((x * x).sum(dim=0) * (y * y).sum(dim=0)).clamp_min(eps)
    return num / den

corr_rg = per_value_corr(r, g)
corr_rb = per_value_corr(r, b)
corr_gb = per_value_corr(g, b)


svd_features = all_tensors[0]["_features_rest"]
svd_features = svd_features.reshape(svd_features.shape[0], svd_features.shape[1]*svd_features.shape[2])
stdevs = torch.std(svd_features,dim=0)
means = torch.mean(svd_features,dim=0)
svd_features= (svd_features-means) / stdevs
U, S, Vh = torch.linalg.svd(svd_features.detach(),full_matrices=False)
print("Singular values:", S)
print("Percentage of variance explained:", S.cumsum(dim=0)/S.sum(dim=0))

with open("corr.csv","w") as f:
    corr_features = all_tensors[0]["_features_rest"].cuda()
    corr_features = corr_features.reshape(corr_features.shape[0], corr_features.shape[1]*corr_features.shape[2])
    stdevs = torch.std(corr_features,dim=0)
    means = torch.mean(corr_features,dim=0)
    print(stdevs.shape)
    print(corr_features.shape)
    corr_features= (corr_features-means) / stdevs
#    for x in range(stdevs.shape[0]):
#        corr_features[:,x] /= stdevs[x]
    pearson = corr_features.T @ corr_features * (1/(corr_features.shape[0]-1))
    for x in range(pearson.shape[0]):
        print(x, end=",",file=f)
    print("",file=f)
    for x in range(pearson.shape[0]):
        for y in range(pearson.shape[1]):
            print(pearson[x,y].item()    ,end=",",file=f)
        print("",file=f)
print("written corr.csv")
import sys
sys.exit(0)
big_correlation_matrix = sh_features.reshape(-1, sh_features.shape[1]*sh_features.shape[2]).T


print("\nPer-value RGB channel correlations:")
print("idx\tcorr(R,G)\tcorr(R,B)\tcorr(G,B)")
for i in range(sh_features.shape[1]):
    print(f"{i:02d}\t{corr_rg[i].item():+.6f}\t{corr_rb[i].item():+.6f}\t{corr_gb[i].item():+.6f}")

def test_algorithm(name,input_sh,*,compress_fn,decompress_fn):
    # input_sh shape: [num_points, 48, 3]
    # inbetween_data shape: ?
    # output_sh shape: [num_points, 48, 3]
    print("Testing algorithm:", name)
    print("------------------")
    inbetween_data = compress_fn(input_sh)
    output_sh = decompress_fn(inbetween_data)

    # print compression ratio
    input_size = input_sh.numel() * input_sh.element_size()
    inbetween_size = inbetween_data.numel() * inbetween_data.element_size()
    compression_ratio = input_size / inbetween_size if inbetween_size > 0 else float('inf')

    print(f"\nCompression Ratio: {compression_ratio:.2f} (input size: {input_size}, compressed size: {inbetween_size})")              

    # For each SH coefficient index, compute the mean absolute difference between the input and output SH coefficients across all points
    abs_diff = torch.abs(input_sh - output_sh)
    mean_abs_diff = abs_diff.mean(dim=0)  # shape: [48, 3]

    print("\nMean Absolute Difference between Input and Output SH Coefficients:")
    print("idx\tmean_abs_diff(R)\tmean_abs_diff(G)\tmean_abs_diff(B)")
    for i in range(input_sh.shape[1]):
        print(f"{i:02d}\t{mean_abs_diff[i, 0].item():.6f}\t{mean_abs_diff[i, 1].item():.6f}\t{mean_abs_diff[i, 2].item():.6f}")

from sklearn.decomposition import PCA

sh_features_normalized = (sh_features - sh_means) / torch.sqrt(sh_variances + 1e-10)

# sh_features_reshaped = sh_features_normalized.reshape(sh_features_normalized.shape[0], sh_features_normalized.shape[1]*sh_features_normalized.shape[2]).detach().cpu().numpy()  # shape: [num_points*48, 3]

# print("Running PCA on SH features...")
# pca = PCA(n_components=48)
# pca.fit(sh_features_reshaped)
# print("\nPCA Explained Variance Ratio:", pca.explained_variance_ratio_)
# print("PCA Explained Variance Ratio (cumulative):", pca.explained_variance_ratio_.cumsum())
# print("PCA Components Shape:", pca.components_.shape)
# # for i in range(min(40, pca.components_.shape[0])):
# #     print(f"Component {i}: {pca.components_[i]}")
    
# import sys
# sys.exit(0)
DEVICE = "cuda"


class Net(torch.nn.Module):
    def __init__(self, feature_shape):
      super(Net, self).__init__()
      feature_count=feature_shape[0]*feature_shape[1]
      self.fc1 = torch.nn.Linear(feature_count, 48*3,device=DEVICE)
      self.fc2 = torch.nn.Linear(48*3, feature_count,device=DEVICE)
      #self.bn = torch.nn.BatchNorm1d(48*3, device=DEVICE)

    def forward(self, x):
        # x=self.bn(x)
        x=self.fc1(x)
        torch.nn.functional.dropout(x, p=0.1, training=self.training)
        x = torch.tanh(x) 
        x=self.fc2(x)
        x = x.reshape(-1, 48, 3)
        return x


class RGBGuessMissingValues:
    """ map RGB components to:
        - RGB DC (3 floats)
        - G SH (47 floats)
        - something else (N floats)
        and then decompress by mapping back to RGB"""
    
    def __init__(self,model):
        self.model = model # nn model to learn mapping from feature zero on each side to the rest of the features


    def compress(self, sh_features):
        # sh_features shape: [num_points, 48, 3]
        # output shape: [num_points, 50]
        in_shape = sh_features.shape
        rval = torch.zeros_like(sh_features)
        # 3D features and DC
        feature_indices_3d = [0,1,3,4,8,9,15]
#        feature_indices_3d = range(0,16)
        features_indices_t = range(16,48)
        for x in feature_indices_3d:
            rval[:,x,:] = sh_features[:,x,:]
        for x in features_indices_t:
            rval[:,x,:] = sh_features[:,x,:]


        return rval.flatten(start_dim=1)

    def decompress(self, compressed_sh):
#        return compressed_sh.reshape(-1, 48, 3)
        # compressed_sh shape: [num_points, 50]
        # output shape: [num_points, 48, 3]#
        rval = self.model(compressed_sh).reshape(-1, 48, 3)

        return rval



all_features = sh_features.to(DEVICE).detach().requires_grad_(True)


#sh_features_normalized.detach().to("cuda").requires_grad_(True)


num_points = all_features.shape[0]
feature_shape = all_features.shape[1:]
test_points = num_points // 10
train_points = num_points - test_points

model = Net(feature_shape)
model = model.to(DEVICE)

compressor = RGBGuessMissingValues(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def weighted_loss(output, target, variances):
    diffs = output - target
    diffs = diffs*diffs
    diffs = torch.mul(diffs, 1.0/(variances+1e-10))
    return torch.mean(diffs)

train_vals = all_features[test_points:]
test_vals = all_features[:test_points]

def shfs_loss(output,target):
    # generate a random set of normalised vector + time diff
    # generate RGBs for the vectors
    randomA = torch.rand((output.shape[0]),device=output.device)*torch.pi
    randomB = torch.rand((output.shape[0]),device=output.device)*2*torch.pi
    time_diffs = torch.normal(0,0.5,(output.shape[0],1),device=output.device)
    vectors = torch.stack([torch.sin(randomA)*torch.cos(randomB), torch.sin(randomA)*torch.sin(randomB), torch.cos(randomA)], dim=1)
    output_rgb = eval_shfs_4d(3,2,output.transpose(2,1),vectors,time_diffs)
    target_rgb = eval_shfs_4d(3,2,target.transpose(2,1),vectors,time_diffs)
    loss = torch.nn.functional.l1_loss(output_rgb, target_rgb) 
    return loss

def shfs_loss_3d(output,target):
    # generate a random set of normalised vector + time diff
    # generate RGBs for the vectors
    print(output.shape,target.shape)
    randomA = torch.rand((output.shape[0]),device=output.device)*torch.pi
    randomB = torch.rand((output.shape[0]),device=output.device)*2*torch.pi
    vectors = torch.stack([torch.sin(randomA)*torch.cos(randomB), torch.sin(randomA)*torch.sin(randomB), torch.cos(randomA)], dim=1)
    output_rgb = eval_sh(3,output[:,:18,:].transpose(2,1),vectors)
    target_rgb = eval_sh(3,target[:,:18,:].transpose(2,1),vectors)
    loss = torch.nn.functional.l1_loss(output_rgb, target_rgb) 
    return loss


@torch.compile
def train():
    #loss_fn = lambda output, target: weighted_loss(output, target, sh_variances.detach())
    #loss_fn = lambda output,target: torch.nn.functional.mse_loss(output, target)
    loss_fn = shfs_loss_3d
    for x in range(1000):
        optimizer.zero_grad()
        compressed = compressor.compress(train_vals)
        output = compressor.decompress(compressed)
        loss = loss_fn(output, train_vals)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            compressed = compressor.compress(test_vals)
            output = compressor.decompress(compressed)
            validation_loss = loss_fn(output, test_vals)
        print(x,loss,validation_loss)

train()

test_algorithm("Uniform",sh_features_normalized.detach().cuda(), compress_fn = compressor.compress, decompress_fn = compressor.decompress)

