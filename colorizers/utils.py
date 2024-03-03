
from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed


def load_img(img_path):
	out_np = np.asarray(Image.open(img_path))
	if(out_np.ndim==2):
		out_np = np.tile(out_np[:,:,None],3)
	return out_np


def resize_img(img, HW=(256,256), resample=3):
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))


# # lab
# def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
# 	# return original size L and resized L as torch Tensors
# 	img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
	
# 	img_lab_orig = color.rgb2lab(img_rgb_orig).transpose(2, 0, 1)
# 	img_lab_rs = color.rgb2lab(img_rgb_rs).transpose(2, 0, 1)

# 	img_l_orig = img_lab_orig[:1,:,:]
# 	img_ab_orig = img_lab_orig[1:,:,:]
# 	img_l_rs = img_lab_rs[:1,:,:]
# 	img_ab_rs = img_lab_rs[1:,:,:]

# 	tens_orig_l = torch.Tensor(img_l_orig)
# 	tens_orig_ab = torch.Tensor(img_ab_orig)
# 	tens_rs_l = torch.Tensor(img_l_rs)
# 	tens_rs_ab = torch.Tensor(img_ab_rs)

# 	return (tens_orig_l, tens_orig_ab, tens_rs_l, tens_rs_ab)


# # lab
# def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
# 	# tens_orig_l 	1 x 1 x H_orig x W_orig
# 	# out_ab 		1 x 2 x H x W

# 	HW_orig = tens_orig_l.shape[2:]
# 	HW = out_ab.shape[2:]

# 	# call resize function if needed
# 	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
# 		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode=mode)
# 	else:
# 		out_ab_orig = out_ab

# 	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
# 	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))


# rgb
def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
	# return original size L and resized L as torch Tensors
	img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
	
	img_rgb_orig = img_rgb_orig.transpose(2, 0, 1) / 255
	img_rgb_rs = img_rgb_rs.transpose(2, 0, 1) / 255

	img_r_orig = img_rgb_orig[:1,:,:]
	img_gb_orig = img_rgb_orig[1:,:,:]
	img_r_rs = img_rgb_rs[:1,:,:]
	img_gb_rs = img_rgb_rs[1:,:,:]

	tens_orig_r = torch.Tensor(img_r_orig)
	tens_orig_gb = torch.Tensor(img_gb_orig)
	tens_rs_r = torch.Tensor(img_r_rs)
	tens_rs_gb = torch.Tensor(img_gb_rs)

	return (tens_orig_r, tens_orig_gb, tens_rs_r, tens_rs_gb)


# rgb
def postprocess_tens(tens_orig_r, out_gb, mode='bilinear'):
	# tens_orig_r 	1 x 1 x H_orig x W_orig
	# out_gb 		1 x 2 x H x W

	HW_orig = tens_orig_r.shape[2:]
	HW = out_gb.shape[2:]

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_gb_orig = F.interpolate(out_gb, size=HW_orig, mode=mode)
	else:
		out_gb_orig = out_gb

	out_rgb_orig = torch.cat((tens_orig_r, out_gb_orig), dim=1)
	return out_rgb_orig.data.cpu().numpy()[0,...].transpose((1,2,0))