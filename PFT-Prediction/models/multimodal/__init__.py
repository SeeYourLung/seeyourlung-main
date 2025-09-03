import functools
from ..attentions import get_attention_module
from .CheXNet import DenseNet121, LightDenseNet121
import torch
import torch.nn as nn
from ..attentions.attention_module import attention_module


model_dict = {
    'densenet121': DenseNet121,
    'lightdensenet121': LightDenseNet121
}

CHEXNET_CKPT_PATH = '/teams/Thymoma_1685081756/PFT/code/WaveletAttention-main/models/CheXNetCKPT/CheXNet.pth.tar'


class MultimodalNet(nn.Module):
    def __init__(self, feature_extractor, image_feature_dim, input_dim_x, output_dim_y, num_slices=20, report_flag=False, report_sentences_num=30, reasoning_flag=False, reasoning_sentences_num=30, ca_num_heads=4, dp_rate=0.2, classification=False):
        super(MultimodalNet, self).__init__()
        self.num_slices = num_slices
        self.report_sentences_num = report_sentences_num
        self.reasoning_sentences_num = reasoning_sentences_num
        self.dp_rate = dp_rate
        self.report_flag = report_flag
        self.reasoning_flag = reasoning_flag
        self.feature_extractor = feature_extractor  # 图像特征提取网络
        self.image_feature_dim = image_feature_dim
        self.res_num = output_dim_y
        self.classification = classification
        if self.report_flag:
            self.ln_report = nn.LayerNorm(self.image_feature_dim)
            self.linear_report = nn.Sequential(
                nn.LayerNorm(769),
                nn.GELU(),
                nn.Linear(769, image_feature_dim),
                nn.LayerNorm(image_feature_dim),
                nn.GELU(),
                nn.Linear(image_feature_dim, 32),
                nn.LayerNorm(32),
                nn.GELU()
            )

        self.relu = nn.ReLU(inplace=False)
        self.gelu = nn.GELU()
        self.ln_img = nn.LayerNorm(self.image_feature_dim)  # need to fix
        self.ln_cov = nn.LayerNorm(16)

        self.linear_cov = nn.Linear(input_dim_x, 16)  # 处理 p 维协变量 x
        self.img_self_att = attention_module(image_feature_dim, image_feature_dim, num_heads=ca_num_heads)

        if self.reasoning_flag:
            self.reason_token_num = 512
            self.linear_reasoning = nn.Sequential(
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Linear(768, image_feature_dim),
                nn.LayerNorm(image_feature_dim),
                nn.GELU()
            )
            self.ln_reason = nn.LayerNorm(self.image_feature_dim)
            self.img2reason_CAs = nn.ModuleList()
            self.reason2img_CAs = nn.ModuleList()
            self.CA_ln_imgs = nn.ModuleList()
            for i in range(self.reasoning_sentences_num):
                self.img2reason_CAs.append(
                    attention_module(image_feature_dim, image_feature_dim, num_heads=ca_num_heads))
                self.reason2img_CAs.append(
                    attention_module(image_feature_dim, image_feature_dim, num_heads=ca_num_heads))
                self.CA_ln_imgs.append(nn.LayerNorm(self.image_feature_dim))

        self.mlp_layers = nn.ModuleList()
        if self.report_flag or self.reasoning_flag:
            self.mlp_input_dim = image_feature_dim + 32 + 16
            if self.reasoning_flag:
                self.conv1d = nn.Conv1d(in_channels=int(self.num_slices * (self.reasoning_sentences_num + 1)), out_channels=1, kernel_size=1)
            else:
                self.conv1d = nn.Conv1d(in_channels=int(self.num_slices), out_channels=1, kernel_size=1)
        else:
            self.mlp_input_dim = image_feature_dim + 16
            self.conv1d = nn.Conv1d(in_channels=int(self.num_slices), out_channels=1, kernel_size=1)

        # self.img_attn1 = nn.Linear(self.modality_num, self.num_slices)
        # self.img_attn2 = nn.Linear(image_feature_dim, 1)
        if self.classification:
            for i in range(1):
                self.mlp_layers.append(nn.Sequential(
                    nn.Linear(self.mlp_input_dim, self.mlp_input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(p=self.dp_rate),
                    nn.Linear(self.mlp_input_dim // 2, self.mlp_input_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(p=(self.dp_rate / 2.0)),
                    nn.Linear(self.mlp_input_dim // 4, output_dim_y),
                    nn.Sigmoid()
                ))
        else:
            for i in range(1):
                self.mlp_layers.append(nn.Sequential(
                    nn.Linear(self.mlp_input_dim, self.mlp_input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(p=self.dp_rate),
                    nn.Linear(self.mlp_input_dim // 2, self.mlp_input_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(p=(self.dp_rate / 2.0)),
                    nn.Linear(self.mlp_input_dim // 4, output_dim_y)
                ))

    def forward(self, image, cov, report=None, reasoning=None):
        # image feature extraction
        batch_size = image.shape[0]
        images = image.view(-1, 1, image.shape[2], image.shape[3])
        images_features = self.feature_extractor(images).view(batch_size, self.num_slices, self.image_feature_dim)
        images_features = self.ln_img(images_features)
        # cov feature extraction
        cov_features = self.ln_cov(self.linear_cov(cov))
        if self.report_flag:
            if self.reasoning_flag:
                images_features, _ = self.img_self_att(images_features, images_features, images_features)  # b*10*128
                # images_features = self.gelu(images_features)

                report_features = self.linear_report(report)  # b*1*32

                reasoning_features = self.linear_reasoning(reasoning)  # b*3*512*128

                # images_features_up_list = []
                images_features_cat = images_features
                for i in range(self.reasoning_sentences_num):
                    # consider CA + CA bi-directional
                    reasoning_features_up, _ = self.reason2img_CAs[i](reasoning_features[:, i, :, :].view(batch_size, -1, self.image_feature_dim), images_features, images_features)
                    images_features_up, _ = self.img2reason_CAs[i](images_features, reasoning_features_up, reasoning_features_up)
                    images_features_up = self.CA_ln_imgs[i](images_features_up + images_features)
                    # images_features_up_list.append(images_features_up)
                    images_features_cat = torch.cat((images_features_cat, images_features_up), dim=1)

                # images_features_up = torch.cat((images_features_up_list[0], images_features_up_list[1], images_features_up_list[2], images_features), dim=1)  # b*40*128

                pooled_images_feature = self.conv1d(images_features_cat).squeeze(1)  # b*128

                pooled_report_feature = report_features.squeeze(1)  # b*32

                mm_features = torch.cat((pooled_images_feature, pooled_report_feature, cov_features), dim=1)   # b*(128+32+16)

                output = self.mlp_layers[0](mm_features) # b*3
            else:
                images_features_up1, _ = self.img_self_att(images_features, images_features, images_features)

                report_features = self.linear_report(report)

                # pooled_images_feature = torch.mean(images_features_up1.transpose(1, 2), dim=-1)
                pooled_images_feature = self.conv1d(images_features_up1).squeeze(1)  # b*128

                pooled_report_feature = report_features.squeeze(1)

                mm_features = torch.cat((pooled_images_feature, pooled_report_feature, cov_features), dim=1)

                output = self.mlp_layers[0](mm_features)

        else:
            images_features_up1, _ = self.img_self_att(images_features, images_features, images_features)

            # pooled_images_feature = torch.mean(images_features_up1.transpose(1, 2), dim=-1)
            pooled_images_feature = self.conv1d(images_features_up1).squeeze(1)

            mm_features = torch.cat((pooled_images_feature, cov_features), dim=1)

            output = self.mlp_layers[0](mm_features)

        # output = self.mlp_layers[0](mm_features)

        return output


def create_multimodal_net(args):
    """
    创建多模态网络，包含图像特征提取网络和协变量处理模块，最后进行回归拟合。
    """

    # 创建图像特征提取Network

    if args.arch.lower() == 'densenet121':
        feature_extractor = model_dict[args.arch.lower()](out_size=args.image_feature_dim)
        checkpoint = torch.load(CHEXNET_CKPT_PATH)
        feature_extractor.load_state_dict(checkpoint['state_dict'], strict=False)
    elif args.arch.lower() == 'lightdensenet121':
        feature_extractor = model_dict[args.arch.lower()](out_size=args.image_feature_dim)

    # 定义完整的多模态网络

    multimodal_net = MultimodalNet(
        feature_extractor=feature_extractor,
        image_feature_dim=args.image_feature_dim,
        input_dim_x=args.cov_dim,  # 协变量的维度
        output_dim_y=args.res_dim,  # 响应变量的维度
        num_slices=args.CT_slice_num,
        report_flag=args.CT_report_flag,
        report_sentences_num=args.CT_report_sentence_num,
        reasoning_flag=args.reasoning_flag,
        reasoning_sentences_num=args.reasoning_sentence_num,
        ca_num_heads=args.CA_num_heads,
        dp_rate=args.dropout,
        classification=args.classification
    )

    return multimodal_net