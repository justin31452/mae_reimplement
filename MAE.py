import torch
import torch.nn as nn
from ViT import *

class MAE(nn.Module):
    def __init__(self, encoder, decoder_dim, 
                 mask_ratio=0.75, decoder_depth=1, 
                 num_decoder_heads=8, decoder_dim_per_head=64):
        super().__init__()

        assert 0. < mask_ratio < 1., f'mask ratio must be kept between 0 and 1, got: {mask_ratio}'
        
        self.encoder = encoder
        self.patch_h, self.patch_w = encoder.patch_h, encoder.patch_w


        num_patches_plus_cls_token, encoder_dim = encoder.pos_embed.shape[-2:]

        num_pixels_per_patch = encoder.patch_embed.weight.size(1)

        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()


        self.mask_ratio = mask_ratio
        self.mask_embed = nn.Parameter(torch.randn(decoder_dim))

        self.decoder = Transformer(
            decoder_dim,
            decoder_dim * 4,
            depth=decoder_depth, 
            num_heads=num_decoder_heads,
            dim_per_head=decoder_dim_per_head, 
        )

        self.decoder_pos_embed = nn.Embedding(num_patches_plus_cls_token - 1, decoder_dim)

        self.head = nn.Linear(decoder_dim, num_pixels_per_patch) 
    
    
    def predict(self, x):
        with torch.no_grad:
            self.eval()

            device = x.device
            b, c, h, w = x.shape

            '''i. Patch partition'''

            num_patches = (h // self.patch_h) * (w // self.patch_w)
            # (b, c=3, h, w)->(b, n_patches, patch_size**2*c)
            patches = x.view(
                b, c,
                h // self.patch_h, self.patch_h, 
                w // self.patch_w, self.patch_w
            ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)

            '''ii. Divide into masked & un-masked groups'''

            num_masked = int(self.mask_ratio * num_patches)

            # Shuffle
            # (b, n_patches)
            shuffle_indices = torch.rand(b, num_patches, device=device).argsort()
            mask_ind, unmask_ind = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]

            # (b, 1)
            batch_ind = torch.arange(b, device=device).unsqueeze(-1)
            mask_patches, unmask_patches = patches[batch_ind, mask_ind], patches[batch_ind, unmask_ind]

            '''iii. Encode'''

            unmask_tokens = self.encoder.patch_embed(unmask_patches)
            # Add position embeddings
            unmask_tokens += self.encoder.pos_embed.repeat(b, 1, 1)[batch_ind, unmask_ind + 1]
            encoded_tokens = self.encoder.transformer(unmask_tokens)

            '''iv. Decode'''
            
            enc_to_dec_tokens = self.enc_to_dec(encoded_tokens)

            # (decoder_dim)->(b, n_masked, decoder_dim)
            mask_tokens = self.mask_embed[None, None, :].repeat(b, num_masked, 1)
            # Add position embeddings
            mask_tokens += self.decoder_pos_embed(mask_ind)

            # (b, n_patches, decoder_dim)
            concat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)
            # dec_input_tokens = concat_tokens
            dec_input_tokens = torch.empty_like(concat_tokens, device=device)
            # Un-shuffle
            dec_input_tokens[batch_ind, shuffle_indices] = concat_tokens
            decoded_tokens = self.decoder(dec_input_tokens)

            '''v. Mask pixel Prediction'''

            dec_mask_tokens = decoded_tokens[batch_ind, mask_ind, :]
            # (b, n_masked, n_pixels_per_patch=patch_size**2 x c)
            pred_mask_pixel_values = self.head(dec_mask_tokens)

            # loss = F.mse_loss(pred_mask_pixel_values, mask_patches)

            # 比较下预测值和真实值
            mse_per_patch = (pred_mask_pixel_values - mask_patches).abs().mean(dim=-1)
            mse_all_patches = mse_per_patch.mean()

            print(f'mse per (masked)patch: {mse_per_patch} mse all (masked)patches: {mse_all_patches} total {num_masked} masked patches')
            print(f'all close: {torch.allclose(pred_mask_pixel_values, mask_patches, rtol=1e-1, atol=1e-1)}')
                
            '''vi. Reconstruction'''

            recons_patches = patches.detach()
            # Un-shuffle (b, n_patches, patch_size**2 * c)
            recons_patches[batch_ind, mask_ind] = pred_mask_pixel_values
            # 模型重建的效果图
            # Reshape back to image 
            # (b, n_patches, patch_size**2 * c)->(b, c, h, w)
            recons_img = recons_patches.view(
                b, h // self.patch_h, w // self.patch_w, 
                self.patch_h, self.patch_w, c
            ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

            mask_patches = torch.randn_like(mask_patches, device=mask_patches.device)
            # mask 效果图
            patches[batch_ind, mask_ind] = mask_patches
            patches_to_img = patches.view(
                b, h // self.patch_h, w // self.patch_w, 
                self.patch_h, self.patch_w, c
            ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

            return recons_img, patches_to_img
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读入图像并缩放到适合模型输入的尺寸
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

# 将图像转换成张量
from torchvision.transforms import ToTensor, ToPILImage

img_ts = ToTensor()(img).unsqueeze(0).to(device)
print(f"input tensor shape: {img_ts.shape} dtype: {img_ts.dtype} device: {img_ts.device}")

# 实例化模型并加载训练好的权重
encoder = ViT(img_size, patch_size, dim=512, mlp_dim=1024, dim_per_head=64)
decoder_dim = 512
mae = MAE(encoder, decoder_dim, decoder_depth=6)
weight = torch.load(('./mae.pth'), map_location='cpu')
mae.to(device)

# 推理
# 模型重建的效果图，mask 效果图
recons_img_ts, masked_img_ts = mae.predict(img_ts)
recons_img_ts, masked_img_ts = recons_img_ts.cpu().squeeze(0), masked_img_ts.cpu().squeeze(0)

# 将结果保存下来以便和原图比较
recons_img = ToPILImage()(recons_img_ts)
recons_img.save('./recons_tree.jpg')

masked_img = ToPILImage()(masked_img_ts)
masked_img.save('./masked_tree.jpg')

