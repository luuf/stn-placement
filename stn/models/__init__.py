import models.mnist as mnist
import models.cifar as cifar
import models.svhn as svhn
import models.plankton as plankton

# dictionaries for mapping arguments to classes

localization_dict = {
    'CNN':      mnist.CNN_localization,
    'CNNmp':    mnist.CNN_mp,
    'CNNCNNmp': mnist.CNNCNN_mp,
    'CNNm':     mnist.CNN_middleloc,
    'CNNmb':    mnist.CNN_middleloc_batchnorm,
    'CNNt':     mnist.CNN_translate,
    'CNNb':     mnist.CNN_localization_batchnorm,
    'CNN2':     cifar.CNN_localization2,
    'FCN':      mnist.FCN_localization,
    'FCNb':     mnist.FCN_localization_batchnorm,
    'FCNmp':    mnist.FCN_localization_maxpool,
    'FCNmpnob': mnist.FCN_maxpool_nobatchnorm,
    'CNNFCN':   mnist.CNNFCN_localization,
    'CNNFCNb':  mnist.CNNFCN_batchnorm,
    'CNNFCNmp': mnist.CNNFCN_maxpool,
    'CNNCNN':   mnist.CNNCNN,
    'ylva':     mnist.ylva_localization,
    'small':    mnist.Small_localization,
    'SVHN-l':   svhn.SVHN_large,
    'SVHN-d':   svhn.SVHN_dropout,
    'SVHN-s':   svhn.SVHN_small,
    'false':    False,
    'plankton-l': plankton.plankton_large,
    'plankton4':  plankton.plankton_depth4,
    'plankton2':  plankton.plankton_depth2,
}

model_dict = {
    'FCN':      mnist.FCN,
    'CNN':      mnist.CNN,
    'CNNb':     mnist.CNN_batchnorm,
    'ylva':     mnist.ylva_mnist,
    'CNN2':     cifar.CNN2,
    'SVHN-CNN': svhn.SVHN_CNN,
    'plankton': plankton.plankton_CNN,
}
