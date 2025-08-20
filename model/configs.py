vgg11_num_layers_list = [1, 1, 2, 2, 2]
vgg13_num_layers_list = [2, 2, 2, 2, 2]
vgg16_num_layers_list = [2, 2, 3, 3, 3]
vgg19_num_layers_list = [2, 2, 4, 4, 4]

resnet18_num_blocks_list=[2, 2, 2, 2]
resnet34_num_blocks_list=[3, 4, 6, 3]
resnet50_num_blocks_list=[3, 4, 6, 3]
resnet101_num_blocks_list=[3, 4, 23, 3]
resnet152_num_blocks_list=[3, 8, 36, 3]

efficientnet_b0_conf = {
    "conv3": {"out_channels": 32},
    "mbconv_blocks": [
        {"out_channels": 16, "kernel_size": 3, "stride": 1, "num_blocks": 1},
        {"out_channels": 24, "kernel_size": 3, "stride": 2, "num_blocks": 2}, 
        {"out_channels": 40, "kernel_size": 5, "stride": 2, "num_blocks": 2},
        {"out_channels": 80, "kernel_size": 3, "stride": 2, "num_blocks": 3},
        {"out_channels": 112, "kernel_size": 5, "stride": 1, "num_blocks": 3}, 
        {"out_channels": 192, "kernel_size": 5, "stride": 2, "num_blocks": 4},
        {"out_channels": 320, "kernel_size": 3, "stride": 1, "num_blocks": 1},
    ],
    "conv1": {"out_channels": 1280}
}

efficientnet_b1_conf = {
    "conv3": {"out_channels": 32},
    "mbconv_blocks": [
        {"out_channels": 16, "kernel_size": 3, "stride": 1, "num_blocks": 2},
        {"out_channels": 24, "kernel_size": 3, "stride": 2, "num_blocks": 3}, 
        {"out_channels": 40, "kernel_size": 5, "stride": 2, "num_blocks": 3},
        {"out_channels": 80, "kernel_size": 3, "stride": 2, "num_blocks": 4},
        {"out_channels": 112, "kernel_size": 5, "stride": 1, "num_blocks": 4}, 
        {"out_channels": 192, "kernel_size": 5, "stride": 2, "num_blocks": 5},
        {"out_channels": 320, "kernel_size": 3, "stride": 1, "num_blocks": 2},
    ],
    "conv1": {"out_channels": 1280}
}

efficientnet_b2_conf = {
    "conv3": {"out_channels": 32},
    "mbconv_blocks": [
        {"out_channels": 16, "kernel_size": 3, "stride": 1, "num_blocks": 2},
        {"out_channels": 24, "kernel_size": 3, "stride": 2, "num_blocks": 3}, 
        {"out_channels": 48, "kernel_size": 5, "stride": 2, "num_blocks": 3},
        {"out_channels": 88, "kernel_size": 3, "stride": 2, "num_blocks": 4},
        {"out_channels": 120, "kernel_size": 5, "stride": 1, "num_blocks": 4}, 
        {"out_channels": 208, "kernel_size": 5, "stride": 2, "num_blocks": 5},
        {"out_channels": 352, "kernel_size": 3, "stride": 1, "num_blocks": 2},
    ],
    "conv1": {"out_channels": 1408}
}

efficientnet_b3_conf = {
    "conv3": {"out_channels": 40},
    "mbconv_blocks": [
        {"out_channels": 24, "kernel_size": 3, "stride": 1, "num_blocks": 2},
        {"out_channels": 32, "kernel_size": 3, "stride": 2, "num_blocks": 3}, 
        {"out_channels": 48, "kernel_size": 5, "stride": 2, "num_blocks": 3},
        {"out_channels": 96, "kernel_size": 3, "stride": 2, "num_blocks": 5},
        {"out_channels": 136, "kernel_size": 5, "stride": 1, "num_blocks": 5}, 
        {"out_channels": 232, "kernel_size": 5, "stride": 2, "num_blocks": 6},
        {"out_channels": 384, "kernel_size": 3, "stride": 1, "num_blocks": 2},
    ],
    "conv1": {"out_channels": 1536}
}