import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import attention

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, alpha=10.0,
                 normalize_input=True,vlad_trigger = True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float 100.0
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = 9
        self.dim = 512
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(self.dim, self.num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(1e-1 * torch.rand(self.num_clusters, self.dim))
        self.encoder_change = nn.Linear(1024,512)


        self._init_params()


    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )
        

    def forward(self, x):
        
        x = x.unsqueeze(-1).permute(0, 2, 1, 3)  #[N, M, dim, 1] -> [N, dim, M, 1]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        x_change = self.encoder_change(x.transpose(1,3)).transpose(1,3)

        N, C = x_change.shape[:2]
        soft_assign = self.conv(x_change).view(N, self.num_clusters, -1)   # [N, K, M]
        soft_assign = F.softmax(soft_assign, dim=1)
        vis_temp = soft_assign.permute(0,2,1)
        # plt.figure(figsize=(6, 8))

        # vis_mat = torch.randn(soft_assign.size(0),49)
        # for i in range(vis_temp.size(0)):
        #     for j in range(49):
        #         vis_mat[i][j] = torch.argmax(vis_temp[i][j])
        # vis_mat = vis_mat/torch.sum(vis_mat,dim=1,keepdim=True)
        # vis_mat = vis_mat.reshape(10,7,7)
        

        x_flatten = x_change.reshape(N, C, -1)

        # calculate residuals to each clusters   [300, num_cluster, 1024, M]
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        # vlad = vlad.view(x.size(0), -1)       # flatten
        # vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        # concat 
        # output = torch.cat((vlad.unsqueeze(-1).transpose(1,2),x_change),dim=2)
        # output_view = output.squeeze(-1).transpose(1,2)
        # attention_mask = (torch.sum(output_view, -1) == 0).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        # return output_view,attention_mask

        # 换成一个attention
        output = torch.cat((vlad,x_flatten.transpose(1,2)),dim=1)
        attention_mask = None
        return output,attention_mask
        # return output,attention_mask,vis_mat