#!/bin/bash
# PyNetIM 权重文件部署脚本

set -e

echo "========================================"
echo "PyNetIM 权重文件部署脚本"
echo "========================================"

# 配置变量
WEIGHTS_SOURCE_DIR="/root/PyNetIM/src/pynetim/algorithms/reinforcement_learning/deep"
WEIGHTS_WEB_DIR="/var/www/pynetim/weights"

# 创建 web 目录
echo ""
echo "1. 创建权重文件目录..."
sudo mkdir -p "$WEIGHTS_WEB_DIR"
sudo chown -R $USER:$USER /var/www/pynetim

# 复制 BiGDN 权重文件
echo ""
echo "2. 复制 BiGDN 权重文件..."
sudo cp "$WEIGHTS_SOURCE_DIR/bigdn/weights/bigdn_weights.pth" "$WEIGHTS_WEB_DIR/"
sudo cp "$WEIGHTS_SOURCE_DIR/bigdn/weights/bigdns_weights.pth" "$WEIGHTS_WEB_DIR/"
sudo cp "$WEIGHTS_SOURCE_DIR/bigdn/weights/node_encoder.pth" "$WEIGHTS_WEB_DIR/"
sudo cp "$WEIGHTS_SOURCE_DIR/bigdn/weights/q_net_s.pth" "$WEIGHTS_WEB_DIR/"

# 复制 ToupleGDD 权重文件
echo ""
echo "3. 复制 ToupleGDD 权重文件..."
sudo cp "$WEIGHTS_SOURCE_DIR/touplegdd/weights/tripling.ckpt" "$WEIGHTS_WEB_DIR/"
sudo cp "$WEIGHTS_SOURCE_DIR/touplegdd/weights/s2vdqn.ckpt" "$WEIGHTS_WEB_DIR/"

# 设置权限
echo ""
echo "4. 设置文件权限..."
sudo chmod 644 "$WEIGHTS_WEB_DIR"/*

# 列出文件
echo ""
echo "5. 权重文件列表:"
ls -lh "$WEIGHTS_WEB_DIR"

echo ""
echo "========================================"
echo "部署完成！"
echo "========================================"
echo ""
echo "权重文件 URL:"
echo "  - https://pynetim.yinjiy.cn/weights/bigdn_weights.pth"
echo "  - https://pynetim.yinjiy.cn/weights/bigdns_weights.pth"
echo "  - https://pynetim.yinjiy.cn/weights/node_encoder.pth"
echo "  - https://pynetim.yinjiy.cn/weights/q_net_s.pth"
echo "  - https://pynetim.yinjiy.cn/weights/tripling.ckpt"
echo "  - https://pynetim.yinjiy.cn/weights/s2vdqn.ckpt"
echo ""
echo "文档页面:"
echo "  - https://pynetim.yinjiy.cn/"
echo ""
echo "验证:"
echo "  curl -I https://pynetim.yinjiy.cn/weights/bigdn_weights.pth"
echo ""
