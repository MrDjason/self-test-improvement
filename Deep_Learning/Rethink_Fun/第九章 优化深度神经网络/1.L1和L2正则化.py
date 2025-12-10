# L1正则化
# Loss L1=Loss(θ)+λ∑i∣θi∣
# L2正则化
# Loss L2=Loss(θ)+λ∑i θi^2

# PyTorch里增加L2正则化
'''
l2_norm = 0.0
for param in model.parameters():
    l2_norm += param.pow(2).sum()
loss = criterion(outputs, labels) + 1e-4 * l2_norm
'''