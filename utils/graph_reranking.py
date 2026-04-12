import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os

def build_global_graph(probFea, galFea, k=20, gamma=0.5):
    """
    Xây dựng Global Graph (Equation 1 áp dụng cho toàn bộ)
    """
    features = torch.cat([probFea, galFea])
    features = F.normalize(features, p=2, dim=1)

    num_feature = features.size(0)
    dist = torch.cdist(features, features, p=2)
    topk_dist, topk_indices = torch.topk(dist, k=k, dim=1, largest=False)

    weights = torch.exp(-(topk_dist ** 2)/ gamma)

    A = torch.zeros((num_feature, num_feature))

    for i in range(num_feature):
        weight_value = weights[i, :k]
        A[i, topk_indices[i, :k]] = weight_value / torch.sum(weight_value)

    return A

def build_cross_camera_graph(probFea, galFea, q_camids, g_camids, k=20, gamma=0.5):
    """
    Xây dựng Cross-Camera Graph (Equation 1 có điều kiện camera khác nhau)
    """
    features = torch.cat([probFea, galFea])
    camids = torch.cat([q_camids, g_camids])

    num_feature = features.size(0)
    dist = torch.cdist(features, features, p=2)
    col = camids.unsqueeze(1)
    row = camids.unsqueeze(0)
    masked_id = (col != row)

    dist[~masked_id] = float('inf')
    topk_dist, topk_indices = torch.topk(dist, k=k, dim=1, largest=False)

    weights = torch.exp(-(topk_dist ** 2)/ gamma)

    A = torch.zeros((num_feature, num_feature))

    for i in range(num_feature):
        weight_value = weights[i, :k]
        A[i, topk_indices[i, :k]] = weight_value / torch.sum(weight_value)

    return A

def normalize_adj(adj_matrix):
    """
    Thực hiện chuẩn hóa: D^(-1/2) * A * D^(-1/2)
    """
    identity_matrix = torch.eye(adj_matrix.shape[0], device=adj_matrix.device)
    adj_matrix_hat = adj_matrix + identity_matrix

    D = adj_matrix_hat.sum(dim=1)
    d_inv_sqrt = torch.pow(D, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

    return d_mat_inv_sqrt @ adj_matrix_hat @ d_mat_inv_sqrt

def graph_reranking(probFea, galFea, q_camids, g_camids, k=20, gamma=0.5, alpha=0.8, learn_based=False, gcn_model=None):
    """
    Hàm chính gọi từ bên ngoài (Main entry point)
    
    Args:
        probFea: Tensor (N, D) - đặc trưng của query
        galFea: Tensor (M, D) - đặc trưng của gallery
        q_camids: Tensor (N,) - ID camera tương ứng của query
        g_camids: Tensor (N,) - ID camera tương ứng của gallery
    
    Returns:
        refined_features: Tensor (N, D) - đặc trưng đã làm sạch
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not isinstance(q_camids, torch.Tensor):
        q_camids = torch.tensor(q_camids, device=device)
    if not isinstance(g_camids, torch.Tensor):
        g_camids = torch.tensor(g_camids, device=device)

    A_global = build_global_graph(probFea, galFea, k, gamma).to(device)
    A_cross = build_cross_camera_graph(probFea, galFea, q_camids, g_camids, k, gamma).to(device)
    
    A_global_norm = normalize_adj(A_global)
    A_cross_norm = normalize_adj(A_cross)
    
    features = torch.cat([probFea, galFea]).to(device)
    if learn_based and gcn_model is not None:
        gcn_model = gcn_model.to(device)
        global_dim = gcn_model.W.shape[0] # Lấy kích thước của Global (768)

        if features.shape[1] > global_dim:
            feat_global = features[:, :global_dim] 
            feat_local  = features[:, global_dim:]

            feat_global_refined = gcn_model(feat_global, A_global_norm, A_cross_norm)  

            feat_global_refined = F.normalize(feat_global_refined, p=2, dim=1)        
            feat_local = F.normalize(feat_local, p=2, dim=1) 

            features = torch.cat((feat_global_refined, feat_local), dim=1)
        else:
            features = torch.mm(features, gcn_model.W)
            features = F.relu(features)
    refined_features = alpha * torch.mm(A_global_norm, features) + \
                       (1 - alpha) * torch.mm(A_cross_norm, features)
    
    num_query = probFea.size(0)
    refined_prob = refined_features[:num_query]
    refined_gal = refined_features[num_query:]

    distmat = torch.cdist(refined_prob, refined_gal, p=2)
           
    return distmat.detach().cpu().numpy() 

class GCNRefiner(nn.Module):
    def __init__(self, feature_dim):
        super(GCNRefiner, self).__init__()

        self.W = nn.Parameter(torch.FloatTensor(feature_dim, feature_dim))
        nn.init.kaiming_uniform_(self.W, a=0.2)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.bn = nn.LayerNorm(feature_dim)
        self.relu = nn.ReLU()

    def forward(self, features, A_global_norm, A_cross_norm):
        alpha = torch.clamp(self.alpha, 0.0, 1.0)

        support = alpha * torch.mm(A_global_norm, features) + \
                  (1 - alpha) * torch.mm(A_cross_norm, features)

        output_gcn = torch.mm(support, self.W)

        output_gcn = self.bn(output_gcn)
        output_gcn = self.relu(output_gcn)

        final_output = features + output_gcn
        
        return final_output
    
    def load_param(self, trained_path):
        if os.path.exists(trained_path):
            state_dict = torch.load(trained_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.load_state_dict(state_dict)
            print(f"==> GCN model loaded from {trained_path}")
        else:
            print(f"==> No GCN checkpoint found at {trained_path}, training from scratch.")