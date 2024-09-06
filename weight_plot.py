import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

#model_name = '/home/mininet/PycharmProjects/pythonProject4/s4_dalition/ablation/checkpoint/0_capnobase__best_model_ablation3_.ckpt'
for root, dirs, files in os.walk("checkpoint"):
    for name in files:
        #model_path = '/home/mininet/PycharmProjects/pythonProject4/s4_dalition/ablation/checkpoint/0_capnobase__best_model_ablation3_.ckpt'
        model_path = '/home/mininet/论文/结果数据/权重文件/0_capnobase__best_model_ablation3_.ckpt'
        #if model_name not in model_path:
            #continue
        weights = torch.load(model_path, map_location=torch.device('cpu'))
        weights_list = {}
        #ll = weights['state_dict']
        #lll = ll['model.freq_module.decoder_module.conv2.weight']
        # energy compaction
        #weights_list['f'] = (1 / np.max((weights['state_dict'])['model.freq_module.decoder_module.conv2.weight'].numpy())) * (weights['state_dict'])['model.freq_module.decoder_module.conv2.weight'].numpy()
        #weights_list['f1'] = ((weights['state_dict'])['model.freq_module.encoder_module.attention.query_projection.weight'].numpy())
        #weights_list['f2'] = ((weights['state_dict'])['model.freq_module.encoder_module.attention.key_projection.weight'].numpy())
        #weights_list['f3'] = ((weights['state_dict'])['model.freq_module.encoder_module.conv2.weight'].numpy())
        #weights_list['f4'] = ((weights['state_dict'])['model.freq_module.decoder_module.conv2.weight'].numpy())
        #weights_list['f5'] = ((weights['state_dict'])['model.freq_module.decoder_module.attention.key_projection.weight'].numpy())
        weights_list['f6'] = ((weights['state_dict'])['model.freq_module.decoder_module.attention.out_projection.weight'].numpy())
        weights_list['f61'] = (1 / np.max((weights['state_dict'])['model.freq_module.decoder_module.attention.out_projection.weight'].numpy())) * (weights['state_dict'])['model.freq_module.decoder_module.attention.out_projection.weight'].numpy()
        weights_list['f7'] = ((weights['state_dict'])['model.decoder2.weight'].numpy())



        weights_list['f31'] = (1 / np.max((weights['state_dict'])['model.freq_module.encoder_module.conv2.weight'].numpy())) * (weights['state_dict'])['model.freq_module.encoder_module.conv2.weight'].numpy()
        #weights_list['t'] = (weights['state_dict'])['model.dilated_block4.conv.weight'].numpy()
        save_root = '/home/mininet/PycharmProjects/pythonProject4/s4_dalition/fig/'

        # 对每个过滤器进行频谱分析


        for w_name, weight in weights_list.items():

            """
            # 获取当前过滤器的权重数组
            filter_weights = weight.squeeze()  # 去除大小为1的维度

            # 对二维数组进行傅立叶变换
            fourier_transform = np.fft.fft2(filter_weights)

            # 计算二维频谱图（幅度谱）
            spectrum = np.abs(fourier_transform)

            # 绘制二维频谱图
            plt.imshow(spectrum, cmap='gray')
            plt.colorbar()
            plt.title('2D Fourier Spectrum')
            plt.xlabel('Frequency (kx)')
            plt.ylabel('Frequency (ky)')
            plt.show()
            """


            fig, ax = plt.subplots()
            im = ax.imshow(weight, cmap='winter')
            fig.colorbar(im, pad=0.03)
            #plt.title(w_name)
            plt.savefig('/home/mininet/PycharmProjects/pythonProject4/s4_dalition/figure/weight.jpg',bbox_inches='tight', pad_inches=0, dpi=2000)  # 注意两个参数
            plt.show()
            #plt.savefig(os.path.join(save_root, w_name + '.png'), dpi=500)
            plt.close()
