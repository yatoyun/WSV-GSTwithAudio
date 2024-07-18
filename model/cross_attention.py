import torch
import torch.nn as nn


class CrossAttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(CrossAttentionModule, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, feature1, feature2):
        query = self.query(feature1)
        key = self.key(feature2)
        value = self.value(feature2)
        
        attention_weights = self.softmax(torch.matmul(query, key.transpose(-2, -1)) / feature1.size(-1)**0.5)
        attended_feature = torch.matmul(attention_weights, value)
        
        return attended_feature

class GatedFeatureFusionWithAttention(nn.Module):
    def __init__(self, feature_dim):
        super(GatedFeatureFusionWithAttention, self).__init__()
        self.cross_attention = CrossAttentionModule(feature_dim)
        self.gate = nn.Linear(feature_dim * 2, feature_dim * 2)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(feature_dim * 2, feature_dim)
    
    def forward(self, feature1, feature2):
        attended_feature2 = self.cross_attention(feature1, feature2) + feature1
        combined_feature = torch.cat((feature1, attended_feature2), dim=-1)
        gate_values = self.sigmoid(self.gate(combined_feature))
        gated_feature = combined_feature * gate_values
        fused_feature = self.fc(gated_feature)
        return fused_feature