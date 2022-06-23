import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from transformers import BertModel
from util import OptimizedF1
from category_id_map import CATEGORY_ID_LIST, cate_l2_num
from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from masklm import MaskLM, MaskVideo, ShuffleVideo
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder

class orgModel(nn.Module):
    def __init__(self, args, pertrain=False):
        super().__init__()
        self.pertrain = pertrain
        self.bert_config = BertConfig.from_pretrained(args.bert_dir)
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        
        self.embeddings = self.bert.embeddings
        self.encoder = self.bert.encoder
        
        self.linear = nn.Linear(768, 768)
        self.video_fc = nn.ReLU()
        self.classifier = nn.Linear(768, len(CATEGORY_ID_LIST))
        self.weight = 0.1
        
        if self.pertrain:
            # mlm
            self.lm = MaskLM(tokenizer_path=args.bert_dir)
            self.num_class = len(CATEGORY_ID_LIST)
            self.vocab_size = self.bert_config.vocab_size
            self.mlmHead = BertOnlyMLMHead(self.bert_config)

            # itm
            self.sv = ShuffleVideo()
            self.newfc_itm = torch.nn.Linear(self.bert_config.hidden_size, 1) 
            
        
    def forward(self, inputs, inference=False):
        
        if self.pertrain:
            
            loss = 0
            
            # mlm
            input_ids, lm_label = self.lm.torch_mask_tokens(inputs['text_input'].cpu())
            text_input_ids = input_ids.to(inputs['text_input'].device)
            lm_label = lm_label[:, 1:].to(inputs['text_input'].device) # [SEP] 卡 MASK 大师 [SEP]

            # itm
            input_feature, video_text_match_label = self.sv.torch_shuf_video(inputs['frame_input'].cpu())
            video_feature = input_feature.to(inputs['frame_input'].device)
            video_text_match_label = video_text_match_label.to(inputs['frame_input'].device)
            
            text_emb = self.text_embeddings(input_ids=text_input_ids)
            video_feature = self.linear(video_feature)
            video_feature = self.video_fc(video_feature)
            video_emb = self.video_embeddings(inputs_embeds=video_feature)
            embedding_output = torch.cat([text_emb[:, 0:1, :], video_emb, text_emb[:, 1:, :]], 1)

            mask_org = torch.cat([inputs['text_mask'][:, 0:1], inputs['frame_mask'], inputs['text_mask'][:, 1:]], 1)
            mask = mask_org[:, None, None, :]
            mask = (1.0 - mask) * -10000.0

            encoder_outputs = self.encoder(embedding_output, attention_mask=mask)['last_hidden_state']
            # features_mean = torch.mean(encoder_outputs, 1)

            # mlm
            mlm_out = self.mlmHead(encoder_outputs)[:, 1 + video_feature.size()[1]: , :]
            pred = mlm_out.contiguous().view(-1, self.vocab_size)
            masked_lm_loss = nn.CrossEntropyLoss()(pred, lm_label.contiguous().view(-1))
            loss += masked_lm_loss / 1.25 / 2
            # itm
            itm_out = self.newfc_itm(encoder_outputs[:, 0, :])
            itm_loss = nn.BCEWithLogitsLoss()(itm_out.view(-1), video_text_match_label.view(-1))
            loss += itm_loss / 2.5 / 2
            
            return loss
        else:
            text_emb = self.embeddings(input_ids=inputs['text_input'])
            video_feature = self.linear(inputs['frame_input'])
            video_feature = self.video_fc(video_feature)
            video_emb = self.embeddings(inputs_embeds=video_feature)
            embedding_output = torch.cat([video_emb, text_emb], 1)

            def eval(embedding_output):
                mask_org = torch.cat([inputs['frame_mask'], inputs['text_mask']], 1)
                # mask_org = (1.0 - mask_org)
                mask = mask_org[:, None, None, :]
                mask = (1.0 - mask) * -10000.0

                encoder_outputs = self.encoder(embedding_output, attention_mask=mask)['last_hidden_state']
                features_mean = torch.mean(encoder_outputs, 1)
                prediction = self.classifier(features_mean)
                return prediction

            smart_loss_fn = SMARTLoss(eval_fn = eval, loss_fn = kl_loss, loss_last_fn = sym_kl_loss)
            prediction = eval(embedding_output)

            if inference:
                # return torch.argmax(prediction, dim=1)
                return prediction
            else:     
                if self.training:
                    loss = self.cal_loss(prediction, inputs['label'])
                    #loss[0] += self.weight * smart_loss_fn(embedding_output, prediction)  
                    #loss[0].requires_grad_(True)
                else:
                    loss = self.cal_loss(prediction, inputs['label'])
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
    
    def calculate_mfm_loss(self, video_feature_output, video_feature_input, 
                           video_mask, video_labels_index, normalize=False, temp=0.1):
        if normalize:
            video_feature_output = torch.nn.functional.normalize(video_feature_output, p=2, dim=2)
            video_feature_input = torch.nn.functional.normalize(video_feature_input, p=2, dim=2)

        afm_scores_tr = video_feature_output.view(-1, video_feature_output.shape[-1])

        video_tr = video_feature_input.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        if normalize:
            logits_matrix = logits_matrix / temp

        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != -100)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss
    
    
def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class VisualPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class VisualLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, 768, bias=False)
        self.bias = nn.Parameter(torch.zeros(768))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
    
