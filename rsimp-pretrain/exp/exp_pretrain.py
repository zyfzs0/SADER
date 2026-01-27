import os
import time
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.plot import visualize_images
import numpy as np
from tqdm import tqdm
import warnings
from thop import profile
import matplotlib.pyplot as plt
from optimizer.muon import SingleDeviceMuonWithAuxAdam

warnings.filterwarnings('ignore')


class Exp_Pretrain(Exp_Basic):
    """
    Pretraining experiment class for image completion tasks.
    Retains original training logic, adds loss recording, CV metrics, and sample visualization.
    """

    def __init__(self, args):
        super(Exp_Pretrain, self).__init__(args)
        self.train_losses = []
        self.val_losses = []

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        dataset, dataloader = data_provider(self.args, flag)
        return dataset, dataloader

    def _select_optimizer(self):
        if self.args.optimizer == 'muon':
            muon_params = []
            adam_params = []

            for n, p in self.model.named_parameters():
                if p.ndim >= 2:
                    if p.requires_grad:
                        muon_params.append(p)
                    else:
                        adam_params.append(p)
                else:
                    adam_params.append(p)

            param_groups = [
                dict(params=muon_params, lr=self.args.learning_rate * 5, momentum=0.95, use_muon=True),
                dict(params=adam_params, lr=self.args.learning_rate, betas=(0.9, 0.95), eps=1e-8, use_muon=False)
            ]

            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
            return optimizer
        elif self.args.optimizer == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight)
        else:
            raise ValueError(f"Unknown optimizer {self.args.optimizer}")

    def _select_criterion(self):
        return nn.L1Loss()

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = 0.0
        time_start = time.time()
        iter_count = 0
        with torch.no_grad():
            for i, (x, mask) in tqdm(enumerate(vali_loader)):
                iter_count += 1
                x = x.float().to(self.device)
                mask = mask.to(self.device)
                outputs = self.model(x, mask)
                mask_expanded = mask.repeat(1, 1, 1, x.shape[-1])
                loss = criterion(x[mask_expanded == 0], outputs[mask_expanded == 0])
                total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(vali_data)
        self.val_losses.append(avg_loss)
        print(f"Validation Loss: {avg_loss:.6f}")
        self.model.train()
        return avg_loss


    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        train_steps = len(train_loader)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            epoch_start = time.time()
            time_start = time.time()
            iter_count = 0

            for i, (x, mask) in tqdm(enumerate(train_loader)):
                iter_count += 1
                x = x.float().to(self.device)
                mask = mask.to(self.device)

                model_optim.zero_grad()
                outputs = self.model(x, mask)
                mask_expanded = mask.repeat(1, 1, 1, x.shape[-1])

                blended = outputs.clone()
                blended[mask_expanded == 1] = x[mask_expanded == 1]

                def gradient(img):
                    dx = img[:, :, 1:, :] - img[:, :, :-1, :]
                    dy = img[:, 1:, :, :] - img[:, :-1, :, :]
                    return dx, dy

                dx_pred, dy_pred = gradient(blended)
                dx_gt, dy_gt = gradient(x)

                boundary_x = (mask_expanded[:, :, 1:, :] != mask_expanded[:, :, :-1, :]).float()
                boundary_y = (mask_expanded[:, 1:, :, :] != mask_expanded[:, :-1, :, :]).float()

                boundary_loss = ((dx_pred - dx_gt) * boundary_x).abs().mean() + ((dy_pred - dy_gt) * boundary_y).abs().mean()

                mask_1ch = mask.permute(0, 3, 1, 2)
                outputs_4d = outputs.permute(0, 3, 1, 2)
                x_4d = x.permute(0, 3, 1, 2)
                C = outputs_4d.shape[1]

                kernel = torch.ones((1, 1, 3, 3), device=outputs.device, dtype=outputs.dtype) / 9.0
                kernel_C = kernel.repeat(C, 1, 1, 1)

                real_sum = F.conv2d(x_4d * mask_1ch, kernel_C, padding=1, groups=C)
                real_count = F.conv2d(mask_1ch, kernel, padding=1)
                real_count_clamped = real_count.clamp(min=1e-6)
                neighborhood_avg = real_sum / real_count_clamped

                weight = real_count
                mask_missing = (1 - mask_1ch)
                mask_missing_C = mask_missing.repeat(1, C, 1, 1)
                weight_C = weight.repeat(1, C, 1, 1)

                smooth_loss = ((torch.abs(outputs_4d - neighborhood_avg) * mask_missing_C) * weight_C).sum() / (mask_missing_C.sum() + 1e-6)

                loss = (
                    criterion(x[mask_expanded == 0], outputs[mask_expanded == 0])
                    + 1.0 * boundary_loss
                    + 1.0 * smooth_loss
                )

                loss.backward()
                model_optim.step()
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0 or (i + 1) == train_steps:
                    iter_time = (time.time() - time_start) / iter_count
                    eta = iter_time * (train_steps - i - 1)
                    print(f"[Epoch {epoch+1}] Step {i+1}/{train_steps} | Loss: {loss.item():.6f} | ETA: {eta:.1f}s")
                    iter_count = 0
                    time_start = time.time()

            avg_train_loss = np.mean(train_loss)
            self.train_losses.append(avg_train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch {epoch+1} completed in {time.time() - epoch_start:.1f}s | "
                f"Train Loss: {avg_train_loss:.6f} | Vali Loss: {vali_loss:.6f} | Test Loss: {test_loss:.6f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            if epoch == 0:
                self.cal_efficiency(x, mask, outputs, setting)

        self._save_loss_plot(path)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        print('loading model')
        checkpoint = torch.load(best_model_path)
        model_keys = set(self.model.state_dict().keys())
        filtered_state_dict = {k: v for k, v in checkpoint.items() if k in model_keys}
        self.model.load_state_dict(filtered_state_dict, strict=False)
        return self.model


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        criterion = self._select_criterion()

        self.model.eval()
        if test:
            print('loading model')
            checkpoint = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))

            model_keys = set(self.model.state_dict().keys())
            filtered_state_dict = {k: v for k, v in checkpoint.items() if k in model_keys}

            self.model.load_state_dict(filtered_state_dict, strict=False)

        total_loss = 0.0
        all_preds = []
        all_targets = []

        result_dir = os.path.join('./results/', setting)
        os.makedirs(result_dir, exist_ok=True)

        time_start = time.time()
        iter_count = 0

        with torch.no_grad():
            for i, (batch_x, masks) in tqdm(enumerate(test_loader)):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                masks = masks.float().to(self.device)
                mask_expanded = masks.repeat(1, 1, 1, batch_x.shape[-1])
                reconstructed = self.model(batch_x, masks)

                loss = criterion(batch_x[mask_expanded == 0],  reconstructed[mask_expanded == 0])
                total_loss += loss.item()

                all_preds.append(reconstructed.cpu().numpy())
                all_targets.append(batch_x.cpu().numpy())

                if i % 20 == 0:
                    visualize_images(
                        original=batch_x*masks,
                        reconstructed=reconstructed,
                        save_path=os.path.join(result_dir, f"sample_{i}.png")
                    )
                    np.save(os.path.join(result_dir, f"pred_{i}.npy"), reconstructed.cpu().numpy())
                    np.save(os.path.join(result_dir, f"targets_{i}.npy"), batch_x.cpu().numpy())
                    np.save(os.path.join(result_dir, f"pred_masked_{i}.npy"), reconstructed[mask_expanded == 0].cpu().numpy())
                    np.save(os.path.join(result_dir, f"targets_masked_{i}.npy"), batch_x[mask_expanded == 0].cpu().numpy())

                    np.save(os.path.join(result_dir, f"masks_{i}.npy"), masks.cpu().numpy())
                    iter_time = (time.time() - time_start) / iter_count
                    eta = iter_time * (len(test_loader) - i - 1)
                    print(f"[Test] Step {i+1}/{len(test_loader)} | Loss: {loss.item():.6f} | ETA: {eta:.1f}s")
                    iter_count = 0
                    time_start = time.time()

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        mae, mse, rmse, psnr, ssim, r2 = metric(all_preds, all_targets)
        avg_loss = total_loss / len(test_loader)

        log_path = os.path.join(self.args.log_dir, self.args.log_name)
        with open(log_path, 'a') as f:
            f.write(f"{setting}\n")
            f.write(f"Test Loss: {avg_loss:.6f}\n")
            f.write(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, R2: {r2:.4f}\n\n")

        print(f"[Test Completed] Avg Loss: {avg_loss:.6f}, "
            f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, R2: {r2:.4f}")



    def cal_efficiency(self, batch_x, batch_mask, outputs, setting):
        """Compute model parameters, FLOPs, and activations."""
        self.model.eval()

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        def count_activations(output):
            return output.numel()

        try:
            macs, params = profile(self.model, inputs=(batch_x, batch_mask), verbose=False)
            flops = macs * 2
            ops = flops
        except Exception:
            macs, params, flops, ops = -1, -1, -1, -1

        total_activations = count_activations(outputs)
        peak_activations = [0]

        def activation_hook(module, inp, out):
            n = out.numel() if isinstance(out, torch.Tensor) else sum(o.numel() for o in out if isinstance(o, torch.Tensor))
            if n > peak_activations[0]:
                peak_activations[0] = n

        hooks = [m.register_forward_hook(activation_hook) for m in self.model.modules() if isinstance(m, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh, nn.Softmax, nn.Conv2d, nn.Linear))]
        _ = self.model(batch_x, batch_mask)
        for h in hooks:
            h.remove()

        os.makedirs(f'./efficiency/{setting}/', exist_ok=True)
        with open(f'./efficiency/{setting}/{self.args.model}.txt', 'w') as f:
            f.write(f"Params (trainable): {count_parameters(self.model)}\n")
            f.write(f"MACs: {macs}\n")
            f.write(f"FLOPs: {flops}\n")
            f.write(f"OPs: {ops}\n")
            f.write(f"Total activations (output): {total_activations}\n")
            f.write(f"Peak activations (single layer): {peak_activations[0]}\n")

        self.model.train()

    def _save_loss_plot(self, save_path):
        plt.figure()
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Curve')
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'loss_curve.png'))
        plt.close()
