import os
import cv2
import time
import tqdm
import numpy as np
import rembg
import copy

import torch
import torch.nn.functional as F

from arguments import ModelParams, PipelineParams, OptimizationParams
from cam_utils import orbit_camera, OrbitCamera, safe_normalize, MiniCam
from scene import GaussianModel
from gaussian_renderer import render

from matplotlib import cm

def easy_cmap(x: torch.Tensor):
    x_rgb = torch.zeros((3, x.shape[0], x.shape[1]), dtype=torch.float32, device=x.device)
    x_max, x_min = 4.0, 0.01
    x_normalize = (x - x_min) / (x_max - x_min)
    x_rgb[0] = torch.clamp(x_normalize, 0, 1)
    x_rgb[1] = torch.clamp(x_normalize, 0, 1)
    x_rgb[2] = torch.clamp(x_normalize, 0, 1)
    return x_rgb

def visualize_depth(depth, near=None, far=None, linear=False):
    colormap = cm.get_cmap('turbo')
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    if linear:
        curve_fn = lambda x: -x
    eps = np.finfo(np.float32).eps
    near = near if near else depth.min()
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]
    out_depth = np.clip(np.nan_to_num(vis), 0., 1.)
    return out_depth

class Trainer:
    def __init__(self, opt, lp, op, pp):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.lp, self.op, self.pp = lp, op, pp
        self.cam = OrbitCamera(opt.ref_size, opt.ref_size, r=opt.radius, fovy=opt.fovy)
        self.seed = "random"
        self.postfix = ''

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None
        self.guidance_svd = None

        self.enable_sd = False
        self.enable_zero123 = False
        self.enable_svd = False

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        self.input_img_list = None
        self.input_mask_list = None
        self.input_img_torch_list = None
        self.input_mask_torch_list = None

        self.N_frames = self.opt.batch_size

        # input text
        self.prompt = ""
        self.negative_prompt = self.opt.negative_prompt

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop

        # load input data from cmdline
        if self.opt.input is not None:  # True
            if opt.data_mode == 'd2':
                self.load_input_d2(self.opt.input)
            else:
                self.load_input(self.opt.input)  # load imgs, if has bg, then rm bg; or just load imgs

        if self.opt.multi_view:
            if opt.data_mode == 'd2':
                self.load_input_m_d2(self.opt.multi_view_input)
            elif opt.data_mode == 'sv4d':
                self.load_input_m_sv4d(self.opt.multi_view_input)
            else:
                self.load_input_m(self.opt.multi_view_input)

        # override prompt from cmdline
        if self.opt.prompt is not None:  # None
            self.prompt = self.opt.prompt

        # renderer
        self.time_duration = [0, (self.N_frames - 1)]
        self.gaussians = GaussianModel(
            sh_degree=self.opt.sh_degree,
            gaussian_dim=4,
            time_duration=self.time_duration,
            rot_4d=True,
        )
        self.gaussain_scale_factor = 1

        # override if provide a checkpoint
        if self.opt.load is not None:
            model_params = torch.load(self.opt.load)
            self.gaussians.restore(model_params, op)
        else:
            # initialize gaussians to a blob
            self.gaussians.initialize(num_pts=self.opt.num_pts)

        self.background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        # print(f'Seed: {seed:d}')
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0

        # setup training
        self.gaussians.training_setup(self.opt)

        # do not do progressive sh-level
        self.gaussians.active_sh_degree = self.gaussians.max_sh_degree
        self.optimizer = self.gaussians.optimizer

        # default camera
        pose = orbit_camera(self.opt.elevation, self.opt.azimuth, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )

        self.enable_sd = self.opt.lambda_sd > 0
        self.enable_zero123 = self.opt.lambda_zero123 > 0
        self.enable_svd = self.opt.lambda_svd > 0 and self.input_img is not None

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:  # False
            if self.opt.mvdream:  # False
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

            for p in self.guidance_sd.parameters():
                p.requires_grad = False

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            self.guidance_zero123 = Zero123(self.device, t_range=[0.02, self.opt.t_max], stable=self.opt.stable_zero123)
            print(f"[INFO] loaded zero123!")

            for p in self.guidance_zero123.parameters():
                p.requires_grad = False

        if self.guidance_svd is None and self.enable_svd:  # False
            print(f"[INFO] loading SVD...")
            from guidance.svd_utils import StableVideoDiffusion
            self.guidance_svd = StableVideoDiffusion(self.device)
            print(f"[INFO] loaded SVD!")

            for p in self.guidance_svd.parameters():
                p.requires_grad = False

        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size),
                                                 mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size),
                                                  mode="bilinear", align_corners=False)

        if self.input_img_list is not None:
            self.input_img_torch_list = [torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(self.device) for
                                         input_img in self.input_img_list]
            self.input_img_torch_list = [
                F.interpolate(input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear",
                              align_corners=False) for input_img_torch in self.input_img_torch_list]

            self.input_mask_torch_list = [torch.from_numpy(input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device) for
                                          input_mask in self.input_mask_list]
            self.input_mask_torch_list = [
                F.interpolate(input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear",
                              align_corners=False) for input_mask_torch in self.input_mask_torch_list]

        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                self.guidance_sd.get_text_embeds(
                    [self.prompt],
                    [self.negative_prompt]
                )

            if self.enable_zero123:
                c_list, v_list = [], []
                for _ in range(self.opt.batch_size):
                    for input_img_torch in self.input_img_torch_list:
                        c, v = self.guidance_zero123.get_img_embeds(input_img_torch)
                        c_list.append(c)
                        v_list.append(v)
                self.guidance_zero123.embeddings = [torch.cat(c_list, 0), torch.cat(v_list, 0)]

            if self.enable_svd:
                self.guidance_svd.get_img_embeds(self.input_img)

    def train_step(self, iteration=0):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.train_steps):  # 1

            self.step += 1  # self.step starts from 0
            step_ratio = min(1, self.step / self.opt.iters)  # 1, step / 500

            # update lr
            self.gaussians.update_learning_rate(self.step)

            loss = 0

            ### known view
            for b_idx in range(self.opt.batch_size):
                cur_cam = copy.deepcopy(self.fixed_cam)
                # cam_time = np.random.randint(0, self.opt.batch_size)
                cam_time = b_idx
                cur_cam.timestamp = cam_time
                outputs = render(cur_cam, self.gaussians, self.pp, self.background)

                w_rgb = 50000
                w_mask = 10000

                # rgb loss
                image = outputs["render"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]
                mask_ref = self.input_mask_torch_list[cam_time].detach()
                loss = loss + w_rgb * F.mse_loss(
                    image,
                    self.input_img_torch_list[cam_time]
                )

                # mask loss
                mask = outputs["alpha"].unsqueeze(0)  # [1, 1, H, W] in [0, 1]
                loss = loss + w_mask * F.mse_loss(
                    mask, self.input_mask_torch_list[cam_time]
                )

                if self.opt.multi_view:
                    azimuths = [
                        8.536657171962021, 15.000000000000000, 28.59643887856473,
                        48.409455846599116, 75.13736283678335,
                        105.000000000000000, 131.59054415340088, 151.40356112143527,
                        164.0880581027295, 171.463342828038,
                        188.536657171962, 195.000000000000000, 208.59643887856473,
                        228.40945584659912, 255.13736283678335,
                        285.000000000000000, 311.59054415340086, 331.40356112143525,
                        344.08805810272946, 351.463342828038,
                        0.0
                    ]

                    subsampled_views_sv4d = np.array(
                        [20, 1, 4, 6, 8, 11, 13, 15, 18]
                    )
                    if self.opt.data_mode == 'syncdreamer':
                        n_mv = 16
                        elevation_mv = -30
                    elif self.opt.data_mode == 'd2':
                        n_mv = 21
                        elevation_mv = 0
                    elif self.opt.data_mode == 'sv4d':
                        n_mv = 9
                        elevation_mv = 0
                    else:
                        n_mv = 0
                        elevation_mv = 0
                    for idx in range(n_mv):
                        if self.opt.data_mode == 'syncdreamer':
                            azimuth = 360/n_mv * idx + self.opt.azimuth
                        elif self.opt.data_mode == 'd2':
                            azimuth = azimuths[idx]
                        elif self.opt.data_mode == 'sv4d':
                            azimuth = azimuths[subsampled_views_sv4d[idx]]

                        if azimuth > 180:
                            azimuth = azimuth - 360
                        pose = orbit_camera(elevation_mv, azimuth, self.opt.radius)  # -30 if syncdreamer
                        cur_cam = MiniCam(
                            pose, 256, 256,
                            self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far,
                            time=cam_time, n_frames=self.N_frames
                        )
                        outputs = render(cur_cam, self.gaussians, self.pp, self.background)

                        # rgb loss
                        image = outputs["render"]  # [3, H, W] in [0, 1]
                        # mask = self.input_masks_torch_list_m[cam_time][idx].detach()
                        loss = loss + self.opt.lambda_mv * F.mse_loss(
                            image,
                            self.input_imgs_torch_list_m[cam_time][idx]
                        )

                        # mask loss
                        mask = outputs["alpha"]  # [1, H, W] in [0, 1]
                        loss = loss + self.opt.lambda_mv * F.mse_loss(mask, self.input_masks_torch_list_m[cam_time][idx])

            ### novel view (manual batch)
            if self.enable_zero123:
                render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
                # render_resolution = 512
                images = []
                poses = []
                vers, hors, radii = [], [], []
                embeddings_c, embeddings_v = [], []
                # avoid too large elevation (> 80 or < -80), and make sure it always cover [-30, 30]
                min_ver = max(min(-30, -30 - self.opt.elevation), -80 - self.opt.elevation)
                max_ver = min(max(30, 30 - self.opt.elevation), 80 - self.opt.elevation)

                for b_idx in range(self.opt.batch_size):
                    cam_time = b_idx
                    for _ in range(1):

                        # render random view
                        ver = np.random.randint(min_ver, max_ver)
                        hor = np.random.randint(-180, 180)
                        radius = 0

                        vers.append(ver)
                        hors.append(hor)
                        radii.append(radius)

                        pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                        poses.append(pose)

                        embeddings_c.append(self.guidance_zero123.embeddings[0][cam_time:cam_time + 1])
                        embeddings_v.append(self.guidance_zero123.embeddings[1][cam_time:cam_time + 1])

                        cur_cam = MiniCam(
                            pose, render_resolution, render_resolution,
                            self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far,
                            time=cam_time, n_frames=self.N_frames
                        )

                        outputs = render(cur_cam, self.gaussians, self.pp, self.background)

                        image = outputs["render"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]
                        images.append(image)

                        # enable mvdream training
                        if self.opt.mvdream:  # False
                            for view_i in range(1, 4):
                                vers.append(ver)
                                hors.append(90 * view_i)
                                radii.append(radius)
                                pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                                poses.append(pose_i)

                                if self.enable_zero123:
                                    embeddings_c.append(self.guidance_zero123.embeddings[0][cam_time:cam_time + 1])
                                    embeddings_v.append(self.guidance_zero123.embeddings[1][cam_time:cam_time + 1])

                                cur_cam_i = MiniCam(
                                    pose_i, render_resolution, render_resolution,
                                    self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far,
                                    time=cam_time, n_frames=self.N_frames
                                )

                                out_i = render(cur_cam_i, self.gaussians, self.pp, self.background)

                                image = out_i["render"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]
                                images.append(image)

                images = torch.cat(images, dim=0)
                poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

                if self.opt.pgc > 0:
                    def _hook(grad):
                        clip_value = self.opt.pgc
                        ratio = 1. / grad.abs() * clip_value
                        ratio[ratio > 1.0] = 1.0
                        grad_ = grad * torch.amin(ratio, dim=[1], keepdim=True)
                        return grad_

                    images.register_hook(_hook)

                # guidance loss
                if self.enable_sd:  # False
                    if self.opt.mvdream:
                        for i in range(self.opt.batch_size):
                            loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(
                                images[i * 4:(i + 1) * 4],
                                poses[i * 4:(i + 1) * 4],
                                step_ratio
                            )
                    else:
                        loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio)

                if self.enable_zero123:
                    embeddings = [torch.cat(embeddings_c, 0), torch.cat(embeddings_v, 0)]
                    strength = step_ratio * 0.45 + 0.5
                    refined_images = self.guidance_zero123.refine(
                        images, vers, hors, radii,
                        strength=strength,
                        embeddings=embeddings
                    ).float()
                    refined_images = F.interpolate(
                        refined_images,
                        (render_resolution, render_resolution),
                        mode="bilinear",
                        align_corners=False
                    )
                    loss = loss + self.opt.lambda_zero123 * F.mse_loss(images, refined_images)

                if self.enable_svd:
                    loss = loss + self.opt.lambda_svd * self.guidance_svd.train_step(images, step_ratio)


            ### regularization loss
            if opt.lambda_point_opa > 0:
                alphas = self.gaussians.get_opacity.clamp(1e-10, 1 - 1e-10)
                # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
                loss_entropy = (-alphas * torch.log(alphas)).mean()
                lambda_point_opa = opt.lambda_point_opa * min(1, 2 * iteration / opt.iters)
                loss = loss + lambda_point_opa * loss_entropy
                print(loss_entropy)

            if opt.lambda_time_smooth > 0:
                time_range = self.time_duration[1] - self.time_duration[0]
                times = torch.linspace(0, time_range, int(30 * time_range)).cuda()
                _, dxyz = self.gaussians.get_current_covariance_and_mean_offset(timestamp=times)
                ddxyz = dxyz[:, 1:, :] - dxyz[:, :-1, :]
                lambda_smooth = opt.lambda_time_smooth * min(1, 2 * iteration / opt.iters)
                loss_smooth = ddxyz.norm(dim=-1).mean()
                loss = loss + lambda_smooth * loss_smooth
                print(loss_smooth)

            # optimize step
            loss.backward()

            # densify and prune
            if self.opt.density_start_iter <= self.step <= self.opt.density_end_iter and self.step < self.opt.iters:
                viewspace_point_tensor, visibility_filter, radii = (
                    outputs["viewspace_points"],
                    outputs["visibility_filter"],
                    outputs["radii"]
                )
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter, self.gaussians._t.grad.clone().detach()
                )

                if self.step % self.opt.densification_interval == 0:
                    self.gaussians.densify_and_prune(
                        self.opt.densify_grad_threshold,
                        min_opacity=0.01, extent=0.5, max_screen_size=1
                    )

            self.optimizer.step()
            self.optimizer.zero_grad()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

    def load_input(self, file_dir, downsample=2):
        from PIL import Image

        crop_size = self.opt.crop_size
        image_size = 256

        def add_margin(pil_img, color=0, size=256):
            width, height = pil_img.size
            result = Image.new(pil_img.mode, (size, size), color)
            result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
            return result

        # file_list = [file_dir.replace('.png', f'_{x:03d}.png') for x in range(self.opt.batch_size)]
        images = sorted(os.listdir(file_dir))
        images.sort(key=lambda x: int(x.split('.')[0]))
        images = [os.path.join(file_dir, image) for image in images]
        num_imgs = len(images)
        if self.opt.data_mode == 'syncdreamer':
            file_list = images[::downsample]
        else:
            file_list = images[:21]
        self.input_img_list, self.input_mask_list = [], []
        for file in file_list:
            # load image
            print(f'[INFO] load image from {file}...')
            image_input = Image.open(file)

            if crop_size != -1:
                img = np.asarray(image_input)
                if img.shape[2] == 3:
                    bg_remover = rembg.new_session()
                    img = rembg.remove(img, session=bg_remover)
                    image_input = Image.fromarray(img)
                alpha_np = np.asarray(image_input)[:, :, 3]
                coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
                min_x, min_y = np.min(coords, 0)
                max_x, max_y = np.max(coords, 0)
                ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
                h, w = ref_img_.height, ref_img_.width
                scale = crop_size / max(h, w)
                h_, w_ = int(scale * h), int(scale * w)
                ref_img_ = ref_img_.resize((w_, h_), resample=Image.BICUBIC)
                image_input = add_margin(ref_img_, size=image_size)
            else:
                # image_input = add_margin(image_input, size=max(image_input.height, image_input.width))
                # image_input = image_input.resize((image_size, image_size), resample=Image.BICUBIC)
                img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                img = img[..., :3]
                img = img[..., ::-1]
                bg_remover = rembg.new_session()
                img = rembg.remove(img, session=bg_remover)
                image_input = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)


            image_input = np.asarray(image_input)
            image_input = image_input.astype(np.float32) / 255.0
            ref_mask = image_input[:, :, 3:]
            image_input[:, :, :3] = image_input[:, :, :3] * ref_mask + 1 - ref_mask  # white background
            self.input_img_list.append(image_input[:, :, :3])
            self.input_mask_list.append(ref_mask)

            # img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            # if img.shape[-1] == 3:
            #     if self.bg_remover is None:
            #         self.bg_remover = rembg.new_session()
            #     img = rembg.remove(img, session=self.bg_remover)
            #     cv2.imwrite(file.replace('.png', '_rgba.png'), img)
            # img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            # img = img.astype(np.float32) / 255.0
            # input_mask = img[..., 3:]
            # # white bg
            # input_img = img[..., :3] * input_mask + (1 - input_mask)
            # # bgr to rgb
            # input_img = input_img[..., ::-1].copy()
            # self.input_img_list.append(input_img)
            # self.input_mask_list.append(input_mask)

        self.input_img_list = self.input_img_list[:self.N_frames]
        self.input_mask_list = self.input_mask_list[:self.N_frames]

    def load_input_m(self, file_dir):
        # load image
        print(f'[INFO] load image from {file_dir}...')
        self.input_imgs_torch_list_m = []
        self.input_masks_torch_list_m = []
        for idx in range(self.opt.batch_size):
            file = os.path.join(file_dir, f'{idx}.png')
            long_img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            input_imgs_m = []
            input_masks_m = []
            image_size = 256
            for index in range(16):
                if self.bg_remover is None:
                    self.bg_remover = rembg.new_session()
                img = rembg.remove(long_img[:, index * image_size:(index + 1) * image_size, :], session=self.bg_remover)
                img = img.astype(np.float32) / 255.0

                mask = img[..., 3:]
                # white bg
                img = img[..., :3] * mask + (1 - mask)
                # bgr to rgb
                img = img[..., ::-1].copy()

                input_img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
                input_img_torch = F.interpolate(input_img_torch, (self.opt.ref_size, self.opt.ref_size),
                                                mode="bilinear", align_corners=False)

                input_mask_torch = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
                input_mask_torch = F.interpolate(input_mask_torch, (self.opt.ref_size, self.opt.ref_size),
                                                 mode="bilinear", align_corners=False)

                input_imgs_m.append(input_img_torch)
                input_masks_m.append(input_mask_torch)

            input_imgs_torch_m = torch.cat(input_imgs_m, dim=0).to(self.device)
            input_masks_torch_m = torch.cat(input_masks_m, dim=0).to(self.device)
            self.input_imgs_torch_list_m.append(input_imgs_torch_m)
            self.input_masks_torch_list_m.append(input_masks_torch_m)


    def load_input_d2(self, file_dir):
        input_img_torch_list = torch.load(
            os.path.join(file_dir, 'ref_video.pth')
        )

        self.input_img_list = []
        self.input_mask_list = []

        for i in range(input_img_torch_list.shape[0]):
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()

            img = input_img_torch_list[i].permute(1, 2, 0).detach().cpu().numpy()
            img = (img * 255).astype(np.uint8)
            img = rembg.remove(img, session=self.bg_remover)
            img = img.astype(np.float32) / 255.0

            mask = img[..., 3:]
            img = img[..., :3] * mask + (1 - mask)
            image_input = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            mask_input = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)

            self.input_img_list.append(image_input)
            self.input_mask_list.append(mask_input[..., None])


    def load_input_m_d2(self, file_dir):
        # load image
        print(f'[INFO] load image from {file_dir}...')
        self.input_imgs_torch_list_m = []
        self.input_masks_torch_list_m = []
        for f_idx in range(self.opt.batch_size):
            input_imgs_m = []
            input_masks_m = []
            for v_idx in range(21):
                file = os.path.join(file_dir, f'view_{v_idx:02d}', f'frame_{f_idx:02d}.png')
                img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                img = rembg.remove(img, session=self.bg_remover)
                img = img.astype(np.float32) / 255.0

                mask = img[..., 3:]
                # white bg
                img = img[..., :3] * mask + (1 - mask)
                # bgr to rgb
                img = img[..., ::-1].copy()

                input_img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
                # input_img_torch = F.interpolate(input_img_torch, (self.opt.ref_size, self.opt.ref_size),
                #                                 mode="bilinear", align_corners=False)

                input_mask_torch = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
                # input_mask_torch = F.interpolate(input_mask_torch, (self.opt.ref_size, self.opt.ref_size),
                #                                  mode="bilinear", align_corners=False)

                input_imgs_m.append(input_img_torch)
                input_masks_m.append(input_mask_torch)

            input_imgs_torch_m = torch.cat(input_imgs_m, dim=0).to(self.device)
            input_masks_torch_m = torch.cat(input_masks_m, dim=0).to(self.device)
            self.input_imgs_torch_list_m.append(input_imgs_torch_m)
            self.input_masks_torch_list_m.append(input_masks_torch_m)


    def load_input_m_sv4d(self, file_dir):
        # load image
        print(f'[INFO] load image from {file_dir}...')
        self.input_imgs_torch_list_m = []
        self.input_masks_torch_list_m = []
        for f_idx in range(self.opt.batch_size):
            input_imgs_m = []
            input_masks_m = []
            for v_idx in range(9):
                file = os.path.join(file_dir, f'view_{v_idx:03d}', f'{f_idx:03d}.png')
                img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                img = rembg.remove(img, session=self.bg_remover)
                img = img.astype(np.float32) / 255.0

                mask = img[..., 3:]
                # white bg
                img = img[..., :3] * mask + (1 - mask)
                # bgr to rgb
                img = img[..., ::-1].copy()

                input_img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
                input_img_torch = F.interpolate(
                    input_img_torch, (self.opt.ref_size, self.opt.ref_size),
                    mode="bilinear", align_corners=False
                )

                input_mask_torch = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
                input_mask_torch = F.interpolate(
                    input_mask_torch, (self.opt.ref_size, self.opt.ref_size),
                    mode="bilinear", align_corners=False
                )

                input_imgs_m.append(input_img_torch)
                input_masks_m.append(input_mask_torch)

            input_imgs_torch_m = torch.cat(input_imgs_m, dim=0).to(self.device)
            input_masks_torch_m = torch.cat(input_masks_m, dim=0).to(self.device)
            self.input_imgs_torch_list_m.append(input_imgs_torch_m)
            self.input_masks_torch_list_m.append(input_masks_torch_m)

    @torch.no_grad()
    def save_model(self):
        os.makedirs(self.opt.outdir, exist_ok=True)
        path = os.path.join(self.opt.outdir)
        torch.save(self.gaussians.capture(), os.path.join(self.opt.outdir, 'model' + self.postfix + '.pth'))

        print(f"[INFO] save model to {path}.")

    # no gui mode
    def train(self, iters=5000):
        from PIL import Image
        import imageio
        from diffusers.utils import export_to_video, export_to_gif

        image_list = []
        depth_list = []

        if iters > 0:
            interval = 1
            nframes = iters // interval
            hor = 180
            delta_hor = 4 * 360 / nframes
            time = 0
            delta_time = 1

            self.prepare_train()

            for i in tqdm.trange(iters):
                self.train_step(i)

                if i % interval == 0:
                    pose = orbit_camera(self.opt.elevation, hor - 180, self.opt.radius)
                    cur_cam = MiniCam(
                        pose, 256, 256,
                        self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far,
                        time=time, n_frames=self.N_frames
                    )
                    with torch.no_grad():
                        outputs = render(cur_cam, self.gaussians, self.pp, self.background)

                    out = outputs["render"].cpu().detach().numpy().astype(np.float32)
                    out = np.transpose(out, (1, 2, 0))
                    out = np.clip(out, 0, 1)
                    out = Image.fromarray(np.uint8(out * 255))
                    image_list.append(out)

                    time = (time + delta_time) % self.N_frames
                    hor = (hor + delta_hor) % 360

            export_to_gif(image_list, f'vis_data/{opt.save_path}/train.gif')

            # render elevation 30
            image_list = []
            depth_list = []
            nframes = self.opt.batch_size
            nviews = 32
            hor = 0
            delta_hor = 360 / nviews
            delta_time = 1
            azimuths = [8.536657171962021, 15.000000000000000, 28.59643887856473, 48.409455846599116, 75.13736283678335,
                        105.000000000000000, 131.59054415340088, 151.40356112143527, 164.0880581027295,
                        171.463342828038,
                        188.536657171962, 195.000000000000000, 208.59643887856473, 228.40945584659912,
                        255.13736283678335,
                        285.000000000000000, 311.59054415340086, 331.40356112143525, 344.08805810272946,
                        351.463342828038,
                        0.0]
            i_azi = 0
            for i_view in tqdm.trange(nviews):
                hor = (hor + delta_hor) % 360
                # hor = azimuths[i_view]
                # if hor > 180:
                #     hor -=360

                time = 0
                for i_time in range(nframes):
                    pose = orbit_camera(-30, hor, self.opt.radius)
                    cur_cam = MiniCam(
                        pose, 512, 512,
                        self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far,
                        time=time, n_frames=self.N_frames
                    )
                    with torch.no_grad():
                        outputs = render(cur_cam, self.gaussians, self.pp, self.background)

                    out = outputs["render"].cpu().detach().numpy().astype(np.float32)
                    out = np.transpose(out, (1, 2, 0))
                    out = np.clip(out, 0, 1)
                    out = Image.fromarray(np.uint8(out * 255))
                    image_list.append(out)

                    # import pdb;pdb.set_trace()
                    # out = outputs["depth"].cpu().detach().numpy().astype(np.float32)
                    out = visualize_depth(easy_cmap(outputs["depth"][0])[0].detach().cpu().numpy(), near=0.1)
                    # out = np.transpose(out, (1, 2, 0))[...,0]
                    # out = np.clip(out, 0, 1)
                    out = Image.fromarray(np.uint8(out * 255))
                    depth_list.append(out)

                    time = (time + delta_time) % self.opt.batch_size

            export_to_gif(image_list, f'vis_data/{opt.save_path}/rgb_ele30.gif')
            export_to_gif(depth_list, f'vis_data/{opt.save_path}/depth_ele30.gif')

            # render elevation 0
            image_list = []
            depth_list = []
            nframes = self.opt.batch_size
            nviews = 32
            hor = 0
            delta_hor = 360 / nviews
            delta_time = 1
            azimuths = [8.536657171962021, 15.000000000000000, 28.59643887856473, 48.409455846599116, 75.13736283678335,
                        105.000000000000000, 131.59054415340088, 151.40356112143527, 164.0880581027295,
                        171.463342828038,
                        188.536657171962, 195.000000000000000, 208.59643887856473, 228.40945584659912,
                        255.13736283678335,
                        285.000000000000000, 311.59054415340086, 331.40356112143525, 344.08805810272946,
                        351.463342828038,
                        0.0]
            i_azi = 0
            for i_view in tqdm.trange(nviews):
                hor = (hor + delta_hor) % 360
                # hor = azimuths[i_view]
                # if hor > 180:
                #     hor -=360

                time = 0
                for i_time in range(nframes):
                    pose = orbit_camera(0, hor, self.opt.radius)
                    cur_cam = MiniCam(
                        pose, 512, 512,
                        self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far,
                        time=time, n_frames=self.N_frames
                    )
                    with torch.no_grad():
                        outputs = render(cur_cam, self.gaussians, self.pp, self.background)

                    out = outputs["render"].cpu().detach().numpy().astype(np.float32)
                    out = np.transpose(out, (1, 2, 0))
                    out = np.clip(out, 0, 1)
                    out = Image.fromarray(np.uint8(out * 255))
                    image_list.append(out)

                    # import pdb;pdb.set_trace()
                    # out = outputs["depth"].cpu().detach().numpy().astype(np.float32)
                    out = visualize_depth(easy_cmap(outputs["depth"][0])[0].detach().cpu().numpy(), near=0.1)
                    # out = np.transpose(out, (1, 2, 0))[...,0]
                    # out = np.clip(out, 0, 1)
                    out = Image.fromarray(np.uint8(out * 255))
                    depth_list.append(out)

                    time = (time + delta_time) % self.opt.batch_size

            export_to_gif(image_list, f'vis_data/{opt.save_path}/rgb_ele0.gif')
            export_to_gif(depth_list, f'vis_data/{opt.save_path}/depth_ele0.gif')

            # save
            self.save_model()


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    lp, op, pp = lp.extract(args), op.extract(args), pp.extract(args)

    os.makedirs(f'vis_data/{opt.save_path}', exist_ok=True)

    trainer = Trainer(opt, lp, op, pp)
    trainer.train(opt.iters)
