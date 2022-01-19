from transformers import AutoModel
from transformers import BertModel, BertPreTrainedModel, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertOnlyMLMHead,BertOnlyNSPHead
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel,RobertaLMHead,RobertaClassificationHead,RobertaForSequenceClassification
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers import logging
import torch.nn.functional as F

logging.set_verbosity_warning()

from transformers.activations import *
from util import *
import copy
class RobertaForMaskedLM(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            # hidden_states=outputs.hidden_states,            
            hidden_states=sequence_output,            
            attentions=outputs.attentions,
        )


class RobertaClassificationHead_SEP(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 3)

    def forward(self, features, sep_ids, **kwargs):
        # x = features[:, 0, :]  # take </s> token (equiv. to [SEP])
        # x = batch (8) * Embedding Size (1024)

        sep_list = [features[ids,sep,:] for ids,sep in enumerate(sep_ids)]
        x = torch.stack(sep_list.copy(), dim=0)
        # print("\nCLS 0 : {}".format(features[0,sep_ids[0],:]))
        # print("CLS 1 : {}".format(features[1,sep_ids[1],:]))
        # print("X 0 : {}".format(x[0]))
        # print("X 1 : {}\n====================\n".format(x[1]))

        # x = batch (8) * Embedding Size (1024)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        sep_list.clear()        
        # x = batch (8) * Label (2)
        return x

class BERT_MLM(BertPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.bert = BertModel(config=config)
        self.lm_head = BertOnlyMLMHead(config)
        self.cls = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        print("\nBERT_MLM\n")
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.predictions.decoder

    def forward(self,
        args=None,        
        input_ids=None, 
        attention_mask=None, 
        token_type_ids=None,
        masked_lm_labels=None,
        labels=None,
        ):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output, pooled_output = output[:2]

        prediction_scores = self.lm_head(sequence_output)
        logits = self.cls(pooled_output)
        
        total_loss = 0
 
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            total_loss += loss

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            total_loss += args.mlm_loss * masked_lm_loss

        return (total_loss,logits)

class RoBERTa_MLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier = RobertaClassificationHead(config)
        print("\nRoBERTa_MLM")
        self.init_weights()

    def forward(self,\
        args=None,\
        input_ids=None,\
        attention_mask=None,\
        masked_lm_labels=None,\
        labels = None,
        ):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)
        logits = self.classifier(sequence_output)

        total_loss = 0
 
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            total_loss += loss

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            total_loss += args.mlm_loss * masked_lm_loss

        return (total_loss,logits)

class RoBERTa_Reformulate(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier = RobertaClassificationHead(config)
        print("\nRoBERTa_Reformulate")
        self.init_weights()

    def forward(self,\
        args=None,\
        input_ids=None,\
        attention_mask=None,\
        masked_lm_labels=None,\
        labels = None,
        ):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)

        # print("1 sequence_output : {}".format(sequence_output.size()))
        # logits = self.classifier(sequence_output)
        # print("1 logits : {}".format(logits.size()))

        sep_token_idx = torch.nonzero(input_ids==2)

        stack = list()
        for num, sep_token in enumerate(sep_token_idx):
            if (num + 1) % 3 == 0:    
                sep_token.data[-1] = sep_token.data[-1] - 1
                dim_0 = sep_token.data[0] 
                dim_1 = sep_token.data[1]
                stack.append(sequence_output[dim_0][dim_1])

        sequence_output = torch.stack(stack.copy())
        sequence_output = sequence_output.unsqueeze(1)
        stack.clear()

        logits = self.classifier(sequence_output)
        total_loss = 0
 
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            total_loss += loss

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            total_loss += args.mlm_loss * masked_lm_loss

        return (total_loss,logits)




class ROBERTA_MLM_weighted_loss(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier = RobertaClassificationHead(config)
        print("\nROBERTA_MLM_weighted_loss")
        self.init_weights()

    def forward(self,\
        args=None,\
        input_ids=None,\
        attention_mask=None,\
        masked_lm_labels=None,\
        labels = None,
        ):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)
        logits = self.classifier(sequence_output)

        total_loss = 0
 
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            total_loss += loss

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_tokens = torch.nonzero(masked_lm_labels!= -100)

            avg_score = list()
            for (bacth_num, pos) in masked_tokens:                
                masked_ids = masked_lm_labels[bacth_num][pos].detach().cpu().numpy().item()
                avg_score.append(args.important_tokens[masked_ids])

            avg_score = sum(avg_score) / len(avg_score)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            # total_loss += args.mlm_loss * masked_lm_loss
            total_loss += avg_score * masked_lm_loss

        return (total_loss,logits)


class RobertaClassificationHead_with_features(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, hidden_size):
        super(RobertaClassificationHead_with_features, self).__init__()
        hidden_size = config.hidden_size + hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, config.num_labels)

    def forward(self, roberta_features, other_model_features, **kwargs):
        x = roberta_features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = torch.cat([x,other_model_features], dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaClassificationHead_hier(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, hidden_size):
        super(RobertaClassificationHead_hier, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



class ROBERTA_MLM_Dummy(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        print("\nROBERTA_MLM_Dummy")        
        self.num_labels = config.num_labels   
        self.classifier = RobertaClassificationHead(config)
        # https://github.com/huggingface/transformers/issues/1234
        # Update config to finetune token type embeddings        
        # self.roberta.config.type_vocab_size = 2

        # # Create a new Embeddings layer, with 2 possible segments IDs instead of 1
        # self.roberta.embeddings.token_type_embeddings = nn.Embedding(2, self.roberta.config.hidden_size)
                        
        # # Initialize it
        # self.roberta.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=self.roberta.config.initializer_range)        


        self.init_weights()

    def forward(self,\
        args=None,\
        input_ids=None,\
        attention_mask=None,\
        token_type_ids=None,\
        masked_lm_labels=None,\
        labels = None,
        ):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)
        logits = self.classifier(sequence_output)

        total_loss = 0
 
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            total_loss += loss

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
#             total_loss += 0.2 * masked_lm_loss
            total_loss += args.mlm_loss*masked_lm_loss

        return (total_loss,logits)


class ROBERTA_MLM_SEP(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels_cls = 2 
        self.num_labels_sep = 3 # Label (3) -> 0: 마스킹된 단어가 없음 / 1: 감성 + 일반 / 2: 일반 단어 
                                # Label (4) -> 0: 마스킹된 단어가 없음 / 1: 감성 단어만 / 2: 감성 + 일반 / 3: 일반 단어 
        self.classifier_CLS = RobertaClassificationHead(config)
        self.classifier_SEP = RobertaClassificationHead_SEP(config)

        print("\nROBERTA_MLM_SEP")

        self.init_weights()

    def get_sep(self,args, input_ids):
        sep_ids = torch.nonzero(input_ids==2)
        sep_ids = sep_ids.detach().cpu().numpy().tolist()
        sep_ids = torch.tensor([value for b_n, value in sep_ids], dtype=torch.long)
        sep_ids.to(args.device)
        return sep_ids

    def forward(self,\
        args=None,\
        input_ids=None,\
        attention_mask=None,\
        masked_lm_labels=None,\
        labels = None,\
        masked_type_labels=None,
        ):

        sep_ids = self.get_sep(args,input_ids)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        total_loss = 0

        if labels is not None:
            logits_cls = self.classifier_CLS(sequence_output)
            loss_fct = nn.CrossEntropyLoss()
            loss_cls = loss_fct(logits_cls.view(-1, self.num_labels_cls), labels.view(-1))
            total_loss += loss_cls
            
        if masked_lm_labels is not None:
            prediction_scores = self.lm_head(sequence_output)
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            total_loss +=  masked_lm_loss

        if masked_type_labels is not None:
            logits_sep = self.classifier_SEP(sequence_output, sep_ids=sep_ids)
            loss_fct = nn.CrossEntropyLoss()
            loss_sep = loss_fct(logits_sep.view(-1, self.num_labels_sep), masked_type_labels.view(-1))
            total_loss += loss_sep

        return (total_loss,logits_cls)


class ROBERTA(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config=config)
        self.num_labels  = config.num_labels = 2        
        self.classifier = RobertaClassificationHead(config)
        self.init_weights()
        print("\nRoBERTa (label : {0})\n".format(config.num_labels))
        
    def forward(self,\
        input_ids=None,\
        attention_mask=None,\
        labels = None,
        ):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(output.last_hidden_state)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss,logits)

class ROBERTA_POST(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.roberta = RobertaForMaskedLM.from_pretrained(pretrained_model_name_or_path=cfg['model'], config=cfg['config'])
        print("\nRoBERTa_POST\n")

    def forward(self,\
        args=None,\
        input_ids=None,\
        attention_mask=None,\
        masked_lm_labels=None,\
        polarity_labels=None,        
        ):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask, labels=masked_lm_labels)
        return (output.loss,)
