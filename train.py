from MAE import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from PIL import Image

img_raw = Image.open('./tree.jpg')
h, w = img_raw.height, img_raw.width
ratio = h / w
print(f"image hxw: {h} x {w} mode: {img_raw.mode}")

img_size, patch_size = (224, 224), (16, 16)
img = img_raw.resize(img_size)
rh, rw = img.height, img.width
print(f'resized image hxw: {rh} x {rw} mode: {img.mode}')
img.save('./resized_tree.jpg')

from torchvision.transforms import ToTensor, ToPILImage

img_ts = ToTensor()(img).unsqueeze(0).to(device)
print(f"input tensor shape: {img_ts.shape} dtype: {img_ts.dtype} device: {img_ts.device}")

encoder = ViT(img_size, patch_size, dim=512, mlp_dim=1024, dim_per_head=64)
decoder_dim = 512
mae = MAE(encoder, decoder_dim, decoder_depth=6)
mae.to(device)

recons_img_ts, masked_img_ts = mae.train(img_ts)
recons_img_ts, masked_img_ts = recons_img_ts.cpu().squeeze(0), masked_img_ts.cpu().squeeze(0)

recons_img = ToPILImage()(recons_img_ts)
recons_img.save('./recons_tree.jpg')

masked_img = ToPILImage()(masked_img_ts)
masked_img.save('./masked_tree.jpg')
