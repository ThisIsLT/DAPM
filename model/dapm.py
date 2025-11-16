import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet50 import ResNet50Encoder
from utils.metric import pose_to_log_depth


# Residual convolutional block
class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        out_ch = out_ch or in_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Dropout2d(0.1)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(self.shortcut, nn.Conv2d):
            nn.init.xavier_uniform_(self.shortcut.weight)

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))




# Linear block with residual connection
class ResidualLinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear[0].weight)
        nn.init.constant_(self.linear[0].bias, 0)
        nn.init.xavier_uniform_(self.linear[4].weight)
        if isinstance(self.shortcut, nn.Linear):
            nn.init.xavier_uniform_(self.shortcut.weight)

    def forward(self, x):
        return F.relu(self.linear(x) + self.shortcut(x))



# Depth Decoder
class DepthDecoderFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        # Lateral convolution + SE module + Dropout
        self.lateral_convs = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, 1),
                    nn.Dropout2d(0.1)  # Add Dropout
                ))
            self.se_blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, max(out_channels//16, 4), 1),
                nn.ReLU(),
                nn.Conv2d(max(out_channels//16, 4), out_channels, 1),
                nn.Sigmoid()
            ))
            self.dropouts.append(nn.Dropout2d(0.1))  # Add Dropout after SE
        
        # FPN convolution with residual connection
        self.fpn_convs = nn.ModuleList([
            ResidualConvBlock(out_channels)  # Residual block replaces normal convolution
            for _ in range(3)
        ])
        
        # Prediction head
        self.depth_head = nn.Sequential(
            ResidualConvBlock(out_channels, out_channels//2),
            nn.Conv2d(out_channels//2, 1, 1),
            nn.Sigmoid()
        )
        
        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x2, x3, x4):
        features = [x2, x3, x4]
        latents = []
        
        # Lateral connection + SE attention + Dropout
        for i in range(3):
            lateral = self.lateral_convs[i](features[i])
            se_weight = self.se_blocks[i](lateral)
            latents.append(self.dropouts[i](lateral * se_weight))
        
        # Feature fusion
        p3 = self.fpn_convs[2](latents[2])
        p2 = self.fpn_convs[1](latents[1] + F.interpolate(p3, scale_factor=2, mode='nearest'))
        p1 = self.fpn_convs[0](latents[0] + F.interpolate(p2, scale_factor=2, mode='nearest'))

        # Depth prediction
        depth_feature = F.interpolate(p1, scale_factor=8, mode='nearest')
        depth = self.depth_head(depth_feature)
        return depth_feature, depth




# Improved PoseDecoder
class PoseDecoder(nn.Module):
    def __init__(self, in_channels_list=[512, 1024, 2048], hidden_dim=256):
        super().__init__()

        # Multi-scale feature processing
        self.d2p_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 64)
        )


        self.p2_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(in_channels_list[0], hidden_dim)
        )
        self.p3_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(in_channels_list[1], hidden_dim)
        )
        self.p4_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(in_channels_list[2], hidden_dim)
        )
        
        # Pose prediction with residual connection
        self.pose_net = nn.Sequential(
            ResidualLinearBlock(hidden_dim*3+64, hidden_dim),
            nn.Linear(hidden_dim, 4),
            # nn.Sigmoid()
        )
        
        # Quantization parameter network
        self.quant_net_16 = nn.Sequential(
            ResidualConvBlock(in_channels_list[2], 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            ResidualLinearBlock(256, 64),
            # nn.Sigmoid()
        )


        self.quant_net_64 = nn.Sequential(
            ResidualConvBlock(in_channels_list[2], 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            ResidualLinearBlock(256, 256),
            # nn.Sigmoid()
        )
        
        # Residual correction network
        self.residual_net_64 = ResidualLinearBlock(64 + 256, 256)
        self.residual_net = ResidualLinearBlock(4 + 64 + 256, 4)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        
        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, d2p, p2, p3, p4):
        # Multi-scale feature fusion
        d2p_feat = self.d2p_net(d2p)

        p2_feat = self.p2_net(p2)
        p3_feat = self.p3_net(p3)
        p4_feat = self.p4_net(p4)
        combined = torch.cat([d2p_feat, p2_feat, p3_feat, p4_feat], dim=1)
        pose = self.pose_net(combined)
        
        # Quantization parameter generation
        quant_16 = self.quant_net_16(p4)
        quant_64 = self.quant_net_64(p4)
        quant_64 = self.residual_net_64(torch.cat([quant_16, quant_64], dim=1))

        init_pose = self.sigmoid(pose)
        
        # Residual correction
        residual_input = torch.cat([pose, quant_16, quant_64], dim=1)
        final_pose = self.residual_net(residual_input)
        final_pose = self.sigmoid(final_pose + pose)  # Residual connection


        quant_16 = quant_16.view(-1, 4, 16)
        quant_64 = quant_64.view(-1, 4, 64)


        return init_pose, final_pose, quant_16, quant_64



# Complete dual-task network
class EfficientDepthPoseNet(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientDepthPoseNet, self).__init__()
        
        # Encoder
        self.depth_encoder = ResNet50Encoder(pretrained)
        
        # Decoder
        self.depth_decoder = DepthDecoderFPN([512, 1024, 2048])
        self.pose_decoder = PoseDecoder([512, 1024, 2048])
        
        # Depth auxiliary module
        self.binary_head = nn.Sequential(
            nn.Conv2d(256+32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        self.quant_head_16 = nn.Sequential(
            nn.Conv2d(256+16+32, 32, 3, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 16, 1),

        )

        self.quant_head_64 = nn.Sequential(
            nn.Conv2d(256+16+64+32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),

        )
        
        self.refine_net = nn.Sequential(
            nn.Conv2d(256+16+64+128+32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )


        self.depth_to_pose = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),  
            nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1), 

        )

        self.binary_down_to16 = nn.Sequential(

            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),  
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), 

        )

        self.p2d_to64 = nn.Sequential(

            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),  
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), 
  
        )

        self.depth_quant_16_to64 = nn.Sequential(

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
        )

        self.depth_quant_64_to128 = nn.Sequential(

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),  
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
        )

        self.pose_to_log_depth = pose_to_log_depth
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        d2, d3, d4 = self.depth_encoder(x)
      
        # Depth decoding
        depth_feature, depth_init = self.depth_decoder(d2, d3, d4)
        d2p = self.depth_to_pose(depth_init)
        init_pose, pose, pose_quant_16, pose_quant_64 = self.pose_decoder(d2p, d2, d3, d4)
        p2d = self.pose_to_log_depth(pose)
        p2d_32 = self.p2d_to64(p2d)


        # Depth auxiliary prediction
        binary_feat = torch.cat([depth_feature, p2d_32], dim=1)
        binary_mask = self.binary_head(binary_feat)
        binary_down = F.adaptive_avg_pool2d(binary_mask, depth_feature.shape[2:])
        binary_down = self.binary_down_to16(binary_down)


        quant_feat_16 = torch.cat([depth_feature, binary_down, p2d_32], dim=1)
        depth_quant_16 = self.quant_head_16(quant_feat_16)
        depth_quant_16_to64 = self.depth_quant_16_to64(depth_quant_16)


        quant_feat_64 = torch.cat([depth_feature, binary_down, depth_quant_16_to64, p2d_32], dim=1)
        depth_quant_64 = self.quant_head_64(quant_feat_64)
        depth_quant_64_to128 = self.depth_quant_64_to128(depth_quant_64)

        
        # Depth refinement
        refine_input = torch.cat([depth_feature, binary_down, depth_quant_16_to64, depth_quant_64_to128, p2d_32], dim=1)
        depth_final = self.refine_net(refine_input) + depth_init + p2d
        depth_final = self.sigmoid(depth_final)
    
        return depth_init, depth_final, init_pose, pose, binary_mask, depth_quant_16, depth_quant_64, pose_quant_16, pose_quant_64, p2d