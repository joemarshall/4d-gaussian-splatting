import time

import torch
torch.set_float32_matmul_precision('high')

from utils.sh_utils import eval_shfs_4d,eval_sh

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
# With SHAC-LST, we have instead:
# 
#












TIMES = []

def add_time(label):
    global TIMES
    last_time = TIMES[-1][2] if len(TIMES)>0 else time.monotonic()
    cur_time = time.monotonic()
    duration = cur_time - last_time
    TIMES.append((label, duration, cur_time))

add_time("start")
all_tensors = torch.load("output/9moving/model_output/chkpnt_resume.pth", map_location="cpu",mmap=True,weights_only = False)
add_time("loaded mmap")
all_tensors = torch.load("output/9moving/model_output/chkpnt_resume.pth", map_location="cpu",mmap=False,weights_only = False)
add_time("Loaded no mmap")
print(TIMES)


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
sh_variances = sh_features.var(dim=0)
print(f"SH Coefficient Means (shape: {sh_features.shape}):")
print(sh_means)
print(f"SH Coefficient Variances (shape: {sh_features.shape}):")
print(sh_variances)



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

