## CHECK FEATURE LAYER!
## Check config settings: Aspect Ratios, clip, scale feature..
# https://github.com/ShuangXieIrene/ssds.pytorch/blob/1233516ad99c95dfb4b5cf1b29a2ced1b8351170/lib/utils/config_parse.py


from layers.prior_box import PriorBox
from ssd import build_ssd
from mobilenet import mobilenet

def create_model():
    '''
    '''
    #
    base = mobilenet
    
    number_box= [2*len(aspect_ratios) if isinstance(aspect_ratios[0], int) else len(aspect_ratios) for aspect_ratios in [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]]]  
        
    model = build_ssd(base=base, feature_layer=[[22, 34, 'S', 'S', '', ''], [512, 1024, 512, 256, 256, 256]], mbox=number_box, num_classes=2)
    #
    feature_maps = _forward_features_size(model, [300,300])
    print('==>Feature map size:')
    print(feature_maps)
    # 
    priorbox = PriorBox(image_size=[300,300], feature_maps=feature_maps, aspect_ratios=[[2,3], [2, 3], [2, 3], [2, 3], [2], [2]], 
                    scale=[0.2, 0.95], archor_stride=[], clip=True)
    # priors = Variable(priorbox.forward(), volatile=True)

    return model, priorbox

def _forward_features_size(model, img_size):
    model.eval()
    x = torch.rand(1, 3, img_size[0], img_size[1])
    x = torch.autograd.Variable(x, volatile=True) #.cuda()
    feature_maps = model(x, phase='feature')
    return [(o.size()[2], o.size()[3]) for o in feature_maps]
