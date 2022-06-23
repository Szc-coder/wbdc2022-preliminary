import torch
from torch import nn
import torch.nn.functional as F

from xbert import BertModel, BertConfig
from timm.models.layers import trunc_normal_, DropPath
from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss

class ALBEF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cls_head = nn.Linear(768, 200)
        self.visual_encoder = TransformerBlock(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=False,
                                              qk_scale=None, drop=0.2, attn_drop=0.2,
                                              drop_path=0.2, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.args = args
        bert_config = BertConfig.from_json_file('bertconfig.json')
        bert_config.num_hidden_layers = 18
        self.text_encoder = BertModel.from_pretrained(args.bert_dir, config=bert_config, add_pooling_layer=False)  
        
        self.share_cross_attention(self.text_encoder.encoder)
        self.weight = 0.1
        
    def forward(self, inputs, inference=False):
        
        frame = inputs['frame_input']
        frame_mask = inputs['frame_mask']
        text = inputs['text_input']
        text_mask = inputs['text_mask']

        image_embeds = self.visual_encoder(frame)
        text_embeds = self.text_encoder.embeddings(input_ids=text)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(frame.device)
        
        embeddings = torch.cat([image_embeds, text_embeds], 1)

        def eval(embeddings):
            
            image_embeds = embeddings[:, 0:32, :]
            text_embeds = embeddings[:, 32:, :]
            
            output = self.text_encoder(inputs_embeds = text_embeds, 
                                       attention_mask = text_mask, 
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,        
                                       return_dict = True,
                                      )  
            hidden_state = output.last_hidden_state[:,0,:]            
            prediction = self.cls_head(hidden_state)
            return prediction
        
        prediction = eval(embeddings)
        smart_loss_fn = SMARTLoss(eval_fn=eval, loss_fn=kl_loss, loss_last_fn=sym_kl_loss)
        
        if inference:
            return prediction
            # return torch.argmax(prediction, dim=1)
        else:     
            loss = self.cal_loss(prediction, inputs['label'])
            if self.args.double_attck:
                loss[0] += self.weight * smart_loss_fn(embeddings, prediction)
            return loss

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        # loss
        loss = F.cross_entropy(prediction, label)
        
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return [loss, accuracy, pred_label_id, label]

    
    def share_cross_attention(self, model):
            
        for i in range(6):
            layer_num = 6+i*2
            modules_0 = model.layer[layer_num].crossattention.self._modules
            modules_1 = model.layer[layer_num+1].crossattention.self._modules

            for name in modules_0.keys():
                if 'key' in name or 'value' in name:
                    module_0 = modules_0[name]
                    module_1 = modules_1[name]
                    if hasattr(module_0, "weight"):
                        module_0.weight = module_1.weight
                        if hasattr(module_0, "bias"):
                            module_0.bias = module_1.bias   

    
    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, register_hook=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, register_hook=False):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
