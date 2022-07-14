import torch
import torch.nn as nn
import torch.optim as optim
import logging

from recommenders.models.newsrec.models.base_model import BaseModel


class FastformerModel(BaseModel):
    def __init__(
        self,
        hparams,
        iterator_creator,
        seed=None,
    ):
        """Initializing the model. Create common logics which are needed by all deeprec models, such as loss function,
        parameter set.

        Args:
            hparams (HParams): A HParams object, holds the entire set of hyperparameters.
            iterator_creator (object): An iterator to load the data.
            graph (object): An optional graph.
            seed (int): Random seed.
        """
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.train_iterator = iterator_creator(
            hparams,
            hparams.npratio,
            col_spliter="\t",
        )
        self.test_iterator = iterator_creator(
            hparams,
            col_spliter="\t",
        )

        self.hparams = hparams
        self.support_quick_scoring = hparams.support_quick_scoring

        # IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
        # Otherwise, their weights will be unavailable in the threads after the session there has been set
        
        self.model = self._build_model()
        self.loss = self._get_loss()
        self.optimizer = self._get_opt()
        
    def _build_model(self):
        """Build Fastformer model.
        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """

        model = Fastformer(self.hparams)

        return model

    def _get_loss(self):
        """Make loss function, consists of data loss and regularization loss
        Returns:
            object: Loss function or loss function name
        """
        if self.hparams.loss == "cross_entropy_loss":
            data_loss = nn.CrossEntropyLoss() 
        else:
            raise ValueError("this loss not defined {0}".format(self.hparams.loss))
        return data_loss

    def _get_opt(self):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            object: An optimizer.
        """
        if optimizer == "adam":
            train_opt = optim.Adam([{'params': self.model.parameters(), 'lr': self.hparams.learning_rate}])

        else:
            raise ValueError("this optimizer not defined {0}".format(self.hparams.loss))

        return train_opt

    def _get_pred(self, logit, task):
        """Make final output as prediction score, according to different tasks.
        Args:
            logit (object): Base prediction value.
            task (str): A task (values: regression/classification)
        Returns:
            object: Transformed score
        """
        if task == "regression":
            pred = tf.identity(logit)
        elif task == "classification":
            pred = tf.sigmoid(logit)
        else:
            raise ValueError(
                "method must be regression or classification, but now is {0}".format(
                    task
                )
            )
        return pred

    def train(self, train_batch_data):
        """Go through the optimization step once with training data in feed_dict.
        Args:
            sess (object): The model session object.
            feed_dict (dict): Feed values to train the model. This is a dictionary that maps graph elements to values.
        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """

        train_input, train_label = self._get_input_label_from_iter(train_batch_data)
        outputs = self.model(train_input)
        loss = self.loss(outputs, train_label)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return outputs, loss

    def eval(self, eval_batch_data):
        """Evaluate the data in feed_dict with current model.
        Args:
            sess (object): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.
        Returns:
            list: A list of evaluated results, including total loss value, data loss value, predicted scores, and ground-truth labels.
        """
        eval_input, eval_label = self._get_input_label_from_iter(eval_batch_data)
        imp_index = eval_batch_data["impression_index_batch"]

        pred_rslt = self.model(eval_input)
        loss = self.loss(pred_rslt, eval_label)

        return pred_rslt, loss.detach().cpu(), eval_label, imp_index

    def run_eval(self, news_filename, behaviors_file):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            filename (str): A file name that will be evaluated.

        Returns:
            dict: A dictionary that contains evaluation metrics.
        """

        if self.support_quick_scoring:
            _, group_labels, group_preds = self.run_fast_eval(
                news_filename, behaviors_file
            )
        else:
            _, group_labels, group_preds = self.run_slow_eval(
                news_filename, behaviors_file
            )
        res = cal_metric(group_labels, group_preds, self.hparams.metrics)
        return res
        
    def fit(
        self,
        train_news_file,
        train_behaviors_file,
        valid_news_file,
        valid_behaviors_file,
        test_news_file=None,
        test_behaviors_file=None,
    ):
        """Fit the model with train_file. Evaluate the model on valid_file per epoch to observe the training status.
        If test_news_file is not None, evaluate it too.
        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            test_news_file (str): test set.
        Returns:
            object: An instance of self.
        """

        self.model.train()

        for epoch in range(1, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch
            epoch_loss = 0
            train_start = time.time()

            tqdm_util = tqdm(
                self.train_iterator.load_data_from_file(
                    train_news_file, train_behaviors_file
                )
            )

            for batch_data_input in tqdm_util:

                _, step_result = self.train(batch_data_input)
                step_data_loss = step_result

                epoch_loss += step_data_loss
                step += 1
                if step % self.hparams.show_step == 0:
                    tqdm_util.set_description(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, epoch_loss / step, step_data_loss
                        )
                    )

            train_end = time.time()
            train_time = train_end - train_start

            eval_start = time.time()

            train_info = ",".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in [("logloss loss", epoch_loss / step)]
                ]
            )

            eval_res = self.run_eval(valid_news_file, valid_behaviors_file)
            eval_info = ", ".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in sorted(eval_res.items(), key=lambda x: x[0])
                ]
            )
            if test_news_file is not None:
                test_res = self.run_eval(test_news_file, test_behaviors_file)
                test_info = ", ".join(
                    [
                        str(item[0]) + ":" + str(item[1])
                        for item in sorted(test_res.items(), key=lambda x: x[0])
                    ]
                )
            eval_end = time.time()
            eval_time = eval_end - eval_start

            if test_news_file is not None:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                    + "\ntest info: "
                    + test_info
                )
            else:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                )
            print(
                "at epoch {0:d} , train time: {1:.1f} eval time: {2:.1f}".format(
                    epoch, train_time, eval_time
                )
            )

        return self

    def group_labels(self, labels, preds, group_keys):
        """Devide labels and preds into several group according to values in group keys.
        Args:
            labels (list): ground truth label list.
            preds (list): prediction score list.
            group_keys (list): group key list.
        Returns:
            list, list, list:
            - Keys after group.
            - Labels after group.
            - Preds after group.
        """

        all_keys = list(set(group_keys))
        all_keys.sort()
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}

        for label, p, k in zip(labels, preds, group_keys):
            group_labels[k].append(label)
            group_preds[k].append(p)

        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])

        return all_keys, all_labels, all_preds

    def run_eval(self, news_filename, behaviors_file):
        """Evaluate the given file and returns some evaluation metrics.
        Args:
            filename (str): A file name that will be evaluated.
        Returns:
            dict: A dictionary that contains evaluation metrics.
        """

        if self.support_quick_scoring:
            _, group_labels, group_preds = self.run_fast_eval(
                news_filename, behaviors_file
            )
        else:
            _, group_labels, group_preds = self.run_slow_eval(
                news_filename, behaviors_file
            )
        res = cal_metric(group_labels, group_preds, self.hparams.metrics)
        return res


class Fastformer(torch.nn.Module):
    """Fastformer model(Additive attention can be all you need)

    Wu, C., Wu, F., Qi, T., Huang, Y., & Xie, X. (2021). Fastformer: 
    Additive attention can be all you need. arXiv preprint arXiv:2108.09084.

    """
    def __init__(self, hparams):
        super(Fastformer, self).__init__()
        self.hparams = hparams
        self.dense_linear = nn.Linear(hparams.hidden_size, hparams.output_size)
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.word_emb_dim, padding_idx=0)
        self.encoder = FastformerEncoder(hparams)
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, input_ids, targets):
        mask = input_ids.bool().float()
        embds = self.word_embedding(input_ids)
        text_vec = self.encoder(embds, mask)
        score = self.dense_linear(text_vec)

        return score

class AttentionPooling(nn.Module):
    def __init__(self, hparams):
        self.hparams = hparams
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(hparams.hidden_size, hparams.hidden_size)
        self.att_fc2 = nn.Linear(hparams.hidden_size, 1)
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
                
    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  
        return x

class FastSelfAttention(nn.Module):
    def __init__(self, hparams):
        super(FastSelfAttention, self).__init__()
        self.hparams = hparams
        if hparams.hidden_size % hparams.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (hparams.hidden_size, hparams.num_attention_heads))
        self.attention_head_size = int(hparams.hidden_size /hparams.num_attention_heads)
        self.num_attention_heads = hparams.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.input_dim= hparams.hidden_size
        
        self.query = nn.Linear(self.input_dim, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key = nn.Linear(self.input_dim, self.all_head_size)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
                
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    

    def forward(self, hidden_states, attention_mask):
        # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
        batch_size, seq_len, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        # batch_size, num_head, seq_len
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / self.attention_head_size**0.5
        # add attention mask
        query_for_score += attention_mask

        # batch_size, num_head, 1, seq_len
        query_weight = self.softmax(query_for_score).unsqueeze(2)

        # batch_size, num_head, seq_len, head_dim
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # batch_size, num_head, head_dim, 1
        pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2).view(-1,1,self.num_attention_heads * self.attention_head_size)
        pooled_query_repeat = pooled_query.repeat(1, seq_len,1)
        # batch_size, num_head, seq_len, head_dim

        # batch_size, num_head, seq_len
        mixed_query_key_layer = mixed_key_layer * pooled_query_repeat
        
        query_key_score = (self.key_att(mixed_query_key_layer)/ self.attention_head_size**0.5).transpose(1, 2)
        
        # add attention mask
        query_key_score += attention_mask

        # batch_size, num_head, 1, seq_len
        query_key_weight = self.softmax(query_key_score).unsqueeze(2)

        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        #query = value
        weighted_value = (pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,))
        weighted_value = self.transform(weighted_value) + mixed_query_layer
      
        return weighted_value

class FastAttention(nn.Module):
    def __init__(self, hparams):
        super(FastAttention, self).__init__()
        self.selfattn = FastSelfAttention(hparams)
        self.output = nn.Sequential(
                                nn.Linear(hparams.hidden_size, hparams.hidden_size),
                                nn.Dropout(hparams.hidden_dropout_prob)
                            )
        self.LayerNorm = nn.LayerNorm(hparams.hidden_size, eps=hparams.layer_norm_eps)

    def forward(self, input_tensor, attention_mask):
        self_output = self.selfattn(input_tensor, attention_mask)
        attention_output = self.output(self_output)
        attention_output = self.LayerNorm(attention_output + input_tensor)

        return attention_output

class FastformerLayer(nn.Module):
    def __init__(self, hparams):
        super(FastformerLayer, self).__init__()
        self.attention = FastAttention(hparams)
        self.intermediate = nn.Sequential(
                                nn.Linear(hparams.hidden_size, hparams.intermediate_size),
                                nn.GELU()
                            )
        self.output = nn.Sequential(
                                nn.Linear(hparams.intermediate_size, hparams.hidden_size),
                                nn.Dropout(hparams.hidden_dropout_prob)
                            )
        self.LayerNorm = nn.LayerNorm(hparams.hidden_size, eps=hparams.layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.LayerNorm(layer_output + attention_output)

        return layer_output
    
class FastformerEncoder(nn.Module):
    def __init__(self, hparams, pooler_count=1):
        super(FastformerEncoder, self).__init__()
        self.hparams = hparams
        self.encoders = nn.ModuleList([FastformerLayer(hparams) for _ in range(hparams.num_hidden_layers)])
        self.position_embeddings = nn.Embedding(hparams.max_position_embeddings, hparams.hidden_size)
        self.LayerNorm = nn.LayerNorm(hparams.hidden_size, eps=hparams.layer_norm_eps)
        self.dropout = nn.Dropout(hparams.hidden_dropout_prob)

        # support multiple different poolers with shared bert encoder.
        self.poolers = nn.ModuleList()
        if hparams.pooler_type == 'weightpooler':
            for _ in range(pooler_count):
                self.poolers.append(AttentionPooling(hparams))
        logging.info(f"This model has {len(self.poolers)} poolers.")

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, 
                input_embs, 
                attention_mask, 
                pooler_index=0):
        #input_embs: batch_size, seq_len, emb_dim
        #attention_mask: batch_size, seq_len, emb_dim

        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype = next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        batch_size, seq_length, emb_dim = input_embs.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device = input_embs.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embs + position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        #print(embeddings.size())
        all_hidden_states = [embeddings]

        for i, layer_module in enumerate(self.encoders):
            layer_outputs = layer_module(all_hidden_states[-1], extended_attention_mask)
            all_hidden_states.append(layer_outputs)
        assert len(self.poolers) > pooler_index
        output = self.poolers[pooler_index](all_hidden_states[-1], attention_mask)

        return output 
    
    
if __name__ == "__main__":
    
    from recommenders.models.newsrec.newsrec_utils import prepare_hparams
    yaml_file = '../hparams/fastformer.yaml'
    batch_size = 32
    epochs = 10

    hparams = prepare_hparams(yaml_file, 
                              batch_size=batch_size,
                              epochs=epochs
                             )

    input_ids, targets = torch.rand(32, 1).long(), torch.rand(32, 1).long()
    m = Fastformer(hparams)
    score = m(input_ids, targets)

    print(score.shape)