import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from math import log2
from torch.nn import init
from torch.autograd import Variable
from models.stylegan_model import ConvLayer, EqualLinear, ConstantInput, StyledConv, ResBlock


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, nc=256, gpu_ids=[0]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class BoundaryLengthGenerator(nn.Module):
    """Generator to calculate the length of editing.
    
    NOTE:
        1. We use two latent codes to generate the editing length, i.e., the original latent
            code and its summation with the unit boundary.
        2. The whole generator is based on the synthesis part of G and also D of StyleGAN, 
            i.e., convert the latent code to a spatial tensor, and using D-like modules.
        3. If ada_len is set to False, the generator will return a fixed length of 5.
            In this case, you CANNOT use identity loss or real fix loss which are expected
            to have zero editing length.
        4. If ada_iter is set to be positive, the length given by the generator
            will be interpolated with a fixed length during [1, ada_iter] 
        5. There are two modes to generate the editing length using ada_iter, i.e., 
            - mode to keep fixed
            - mode to edit
           If you do not use ada_iter, then both two modes are generated automatically
            by the generator.
    """

    def __init__(self, 
                 spatial_dim=8, 
                 fix_len=5,
                 ada_len=False,
                 ada_iter=-1,
                 max_len=10,
                 latent_dim=512):
        """Initialize the length generator.
        
        Args:
            spatial_dim (int)  --  spatial dimension in the first conv of StyleGAN
            fix_len (int)      --  fixed editing length
            ada_len (bool)     --  whether to use adaptive editing length given by the generator
            ada_iter (int)     --  how many iterations to use interpolation with generated length and fixed length, default to be -1 to avoid interpolation
            max_len (int)      --  the maximum of the generated editing length
            latent_dim (int)   --  dimension of style code
        """
        
        super(BoundaryLengthGenerator, self).__init__()
        self.spatial_dim = spatial_dim
        self.fix_len = fix_len
        self.ada_len = ada_len
        self.ada_iter = ada_iter
        num_channels = 512
        self.max_len = max_len

        # same as the conv1 in StyleGAN2
        self.input = ConstantInput(num_channels, spatial_dim)
        self.conv1 = StyledConv(num_channels, num_channels, 3, latent_dim)

        # modules similar to the D network of StyleGAN2
        # combine the the results from the two latent codes together
        # then output with the shape (bs, num_channels, 2, 2)
        convs = [ResBlock(num_channels * 2, num_channels)]
        for i in range(int(log2(spatial_dim)), 2, -1):
            convs += [ResBlock(num_channels, num_channels)]
        convs += [ConvLayer(num_channels, num_channels, 3)]
        self.convs = nn.Sequential(*convs)

        # linear block and the final predicting scores
        final_layer = [
            EqualLinear(num_channels * 2 * 2, num_channels, activation="fused_lrelu"),
            EqualLinear(num_channels, 1),
        ]
        self.final_layer = nn.Sequential(*final_layer)

    def forward(self, latent, len_type, boundary, iter, fix_flag=False):
        """Calculate the editing length given the latent code and the boundary.
        
        NOTE:
            1. You CANNOT simultaneously set fix_flag=True and len_type='fix', i.e., when using fixed length
                of editing, you may not expect the image to be kept fixed
            
            2. If need to do the interpolation, use `fix_len` if not `fix_flag`, use 0 if `fix_flag`.
                Also, you are NOT allowed to use len_type='ada_interp' when testing where iter will be -1.

            3. You may use all three len_type when ada_len=True, however, you can only use
                len_type='fix' when ada_len=False
        
        Args:
            latent (torch.Tensor)   --  the given latent code based on which to calculate the length
            len_type (str)          --  use which type of length calculation, [ fix | ada | ada_interp ]
            boundary (torch.Tensor) --  the given boundary based on which to calculate the length
            iter (int)              --  the present iteration count, use for interpolation
            fix_flag (bool)         --  which mode to use to edit the image
        """
        
        bs = latent.shape[0]

        # use fixed length, directly return
        if len_type == 'fix':
            assert not fix_flag, f'You CANNOT set fix_flag to be [{fix_flag}] when using fixed length (ada_len={self.ada_len}).'
            return self.fix_len * torch.ones(bs, 1, device=latent.device)

        # if len_type is not `fix`, first calculate the length
        elif len_type in ['ada', 'ada_interp']:
            assert self.ada_len, f'You can ONLY use fixed length when [ada_len] is closed.'
            
            # the same pipeline as StyleGAN2
            latent1 = latent
            latent2 = latent + F.normalize(boundary, dim=1)
            out = self.input(latent)
            out1 = self.conv1(out, latent1)
            out2 = self.conv1(out, latent2)

            # concat
            x = self.convs(torch.cat([out1, out2], dim=1))
            x = self.final_layer(x.view(bs, -1))

            # normalize to [0, 1]
            length = torch.sigmoid(x) * self.max_len

            # use interpolation
            if len_type == 'ada_interp':
                assert self.ada_iter > 0, f'To use [ada_interp], you are required to set [ada_iter] (now: {self.ada_iter}) > 0.'
                assert 1 <= iter <= self.ada_iter, f'The only allowed interval of iteration is [1, {self.ada_iter}] while TRAINING.'
                target_len = 0 if fix_flag else self.fix_len
                ratio = iter / float(self.ada_iter)
                length = length * ratio + target_len * (1 - ratio)

        else:
            raise NotImplementedError

        return length


class BoundaryGenerator(nn.Module):
    """Generator to train boundary for editing.
    
    NOTE:
        1. We do not have any restriction or normalization for the length of the 
            boundary which is a parameter of the whole network. Hence we need to
            divide the length generated by BoundaryLengthGenerator by the norm
            of the boundary.
        
        2. if stop updating boundary, all calculation will use the detached version
            of boundary.

    TODO:
        1. When dividing by the norm of the boundary, if need to use detach? On
            toy model, training with boundary groundtruth and hence paired images
            or latent codes, it is better not to detach (boundary norm and loss
            are smaller).
    """

    def __init__(self, 
                 latent_dim=512, 
                 init_boundary=None,
                 fix_len=5,):
        """Initialize the boundary generator.
        
        Args:
            latent_dim (int)              --  dimension of style code
            init_boundary (torch.Tensor)  --  a pretrained boundary for easy training
            fix_len (int)                 --  fixed editing length
        """
        super(BoundaryGenerator, self).__init__()
        self.latent_dim = latent_dim 
        self.init_boundary = init_boundary
        self.fix_len = fix_len
        # use the given `init_boundary` or generate randomly
        boundary = F.normalize(torch.randn(1, latent_dim), dim=1) if init_boundary is None else init_boundary
        self.boundary = nn.Parameter(boundary, requires_grad=True)

    def forward(self, latent, length=None, cond_boundary=None):
        """Calculate the editing boundary given the latent code.
        
        NOTE:
            1. You CANNOT simultaneously set fix_flag=True and len_type='fix', i.e., when using fixed length
                of editing, you may not expect the image to be kept fixed
            
            2. If need to do the interpolation, use `fix_len` if not `fix_flag`, use 0 if `fix_flag`.
                Also, you are NOT allowed to use len_type='ada_interp' when testing.

            3. You may use all three len_type when ada_len=True, however, you can only use
                len_type='fix' when ada_len=False

            4. when setting len_type='default':

                case 1: ada_len = False                             -->  len_type = 'fix'
                
                case 2: ada_len = True:
                    case 2.1: iter <= 0 (default value)             -->   len_type = 'ada'
                    
                    case 2.2: iter > 0 (when training):
                        case 2.2.1: ada_iter <= 0 (default value)   -->   len_type = 'ada'
                        case 2.2.2: ada_iter>0 and iter>ada_iter    -->   len_type = 'ada'
                        case 2.2.3: ada_iter>0 and iter<=ada_iter   -->   len_type = 'ada_interp'
        
        Args:
            latent (torch.Tensor)         --  the given latent code based on which to calculate the length
            len_type (str)                --  use which type of length calculation, [ default | fix | ada | ada_interp ]
            fix_flag (bool)               --  which mode to use to edit the image
            iter (int)                    --  the present iteration count, whether to stop updating boundary
            cond_boundary (torch.Tensor)  --  boundary for conditional editing
            use_fixed_len (bool)          --  simply use `self.fix_len` as length, only for calculation of G cl loss
        """

        # calculate the editing length
        # when calculating editing length, detach boundary, see TODO above
        if length is None:
            length = self.fix_len / self.boundary.detach().norm(dim=1)
        else:
            length = length / self.boundary.detach().norm(dim=1)

        # edited latent code
        latent_edited = latent + self.boundary * length

        # if doing conditional editing, simply do the projection
        if cond_boundary is None:
            return latent_edited
        v = latent_edited - latent
        v_proj = v - torch.einsum('ik,jk->ij', [v, cond_boundary]) * cond_boundary
        latent_edited = latent + v_proj
        return latent_edited


class Classifier(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.channel_multiplier = 2
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * self.channel_multiplier,
            128: 128 * self.channel_multiplier,
            256: 64 * self.channel_multiplier,
            512: 32 * self.channel_multiplier,
            1024: 16 * self.channel_multiplier,
        }
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, logits):
        bs = logits.shape[0]
        out = self.final_linear(logits.view(bs, -1))
        return out


class ContrastiveLoss(nn.Module):
    def __init__(self, hidden_channels=512, queue_size=1000):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.temperature = 0.2
        self.queue_size = queue_size

        self.mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, 128))
        
        self.register_buffer("queue", torch.randn(128, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        keys = keys.T
        ptr = int(self.queue_ptr) 
        if batch_size > self.queue_size:
            self.queue[:, 0:] = keys[:, :self.queue_size]

        elif ptr + batch_size > self.queue_size:
            self.queue[:, ptr:] = keys[:, :self.queue_size - ptr]
            self.queue[:, :batch_size - (self.queue_size - ptr)] = keys[:, self.queue_size-ptr:]
            self.queue_ptr[0] = batch_size - (self.queue_size - ptr)
        else:
            self.queue[:, ptr:ptr + batch_size] = keys
            self.queue_ptr[0] = ptr + batch_size

    def forward(self, feats_q, feats_kp, feats_km):
        if feats_q.ndim > 2:
            feats_q = feats_q.mean([2, 3])
        q = self.mlp(feats_q)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            if feats_kp.ndim > 2:
                feats_kp = feats_kp.mean([2, 3])
            kp = self.mlp(feats_kp)
            kp = F.normalize(kp, dim=1)
            if feats_km.ndim > 2:
                feats_km = feats_km.mean([2, 3])
            km = self.mlp(feats_km)
            km = F.normalize(km, dim=1)
 
        l_pos = torch.einsum('nc,nc->n', [q, kp]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], 1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        
        self._dequeue_and_enqueue(km)
        return loss


class InsCLLoss(nn.Module):
    def __init__(self, hidden_channels=256, temperature=0.2, queue_size=3500, momentum=0.999):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.temperature = temperature
        self.queue_size = queue_size
        self.momentum = momentum

        self.mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, 128))
        self.momentum_mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, 128))
        self.momentum_mlp.requires_grad_(False)

        for param_q, param_k in zip(self.mlp.parameters(), self.momentum_mlp.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(128, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.mlp.parameters(), self.momentum_mlp.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        keys = keys.T
        ptr = int(self.queue_ptr) 
        if batch_size > self.queue_size:
            self.queue[:, 0:] = keys[:, :self.queue_size]

        elif ptr + batch_size > self.queue_size:
            self.queue[:, ptr:] = keys[:, :self.queue_size - ptr]
            self.queue[:, :batch_size - (self.queue_size - ptr)] = keys[:, self.queue_size-ptr:]
            self.queue_ptr[0] = batch_size - (self.queue_size - ptr)
        else:
            self.queue[:, ptr:ptr + batch_size] = keys
            self.queue_ptr[0] = ptr + batch_size

    def forward(self, im_q, im_k, loss_only=False, update_q=False):
        device = im_q.device

        # compute query features and project to an unit ball
        if im_q.ndim > 2:
            im_q = im_q.mean([2, 3])
        q = self.mlp(im_q)
        q = F.normalize(q, dim=1)

        # compute key features with NO gradient and project to an unit ball
        with torch.no_grad():
            self._momentum_update_key_encoder()
            if im_k.ndim > 2:
                im_k = im_k.mean([2, 3])
            k = self.momentum_mlp(im_k)
            k = F.normalize(k)
        
        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # N*1
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # N*K
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(device)

        if not loss_only:
            if update_q:
                self._dequeue_and_enqueue(q)
            else:
                self._dequeue_and_enqueue(k)

        # calculate loss
        loss = F.cross_entropy(logits, labels)

        return loss


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
        self.fc = nn.Linear(2048 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


def resnet50(num_classes):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=Bottleneck, 
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=False)
    return model
