import copy
from typing import Optional, Any

import torch, pickle
from torch import Tensor,nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from torch.autograd import Variable

from torch.nn.parameter import Parameter

from typing import Optional, Tuple, Union
from math import *
from modules.activation import HopfieldCore
from Hyperparameters import args
import numpy as np

class Transformer(Module):
    r"""A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).

    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None) -> None:
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`.
            - tgt: :math:`(T, N, E)`.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, N, E)`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output


    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)



class EnergyTransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(EnergyTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, training = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        prev_out = None

        Energy = 0
        Error = torch.Tensor([0]).to(args['device'])
        for mod in self.layers:
            output, E = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, training = training)
            Energy += E
            if prev_out is None:
                prev_out = output
            else:
                prev_out = prev_out + F.dropout(output)
                prev_out = self.norm(prev_out)

        if self.norm is not None:
            output = self.norm(output)

        return output, Energy, Error



class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class EnergyTransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_stored_pattern: bool = True,
                 normalize_stored_pattern_affine: bool = True,
                 normalize_state_pattern: bool = True,
                 normalize_state_pattern_affine: bool = True,
                 normalize_pattern_projection: bool = True,
                 normalize_pattern_projection_affine: bool = True,
                 hidden_size: Optional[int] = None,
                 output_size: Optional[int] = None,
                 pattern_size: Optional[int] = None,
                 scaling: Optional[Union[float, Tensor]] = None,

                 update_steps_max: Optional[Union[int, Tensor]] = 0,
                 update_steps_eps: Union[float, Tensor] = 1e-4,
                 num_heads: int = 1,
                 normalize_hopfield_space: bool = False,
                 normalize_hopfield_space_affine: bool = False,
                 stored_pattern_as_static: bool = False,
                 state_pattern_as_static: bool = False,
                 pattern_projection_as_static: bool = False,
                 pattern_projection_as_connected: bool = False,
                 stored_pattern_size: Optional[int] = None,
                 pattern_projection_size: Optional[int] = None,

                 batch_first: bool = True,
                 association_activation: Optional[str] = None,
                 input_bias: bool = True,
                 concat_bias_pattern: bool = False,
                 add_zero_association: bool = False,
                 disable_out_projection: bool = False):
        super(EnergyTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.W = xavier_uniform_(torch.rand([d_model, d_model]))
        self.W = Parameter(self.W)
        self.all_attn_linear = Linear(d_model, 1)
        self.linear_reweight = Linear(d_model, d_model)
        self.linear_all2= Linear(d_model, d_model)
        self.activation = _get_activation_fn(activation)
        self.association_core = HopfieldCore(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, bias=input_bias,
            add_bias_kv=concat_bias_pattern, add_zero_attn=add_zero_association, kdim=stored_pattern_size,
            vdim=pattern_projection_size, head_dim=hidden_size, pattern_dim=pattern_size, out_dim=output_size,
            disable_out_projection=disable_out_projection, key_as_static=stored_pattern_as_static,
            query_as_static=state_pattern_as_static, value_as_static=pattern_projection_as_static,
            value_as_connected=pattern_projection_as_connected, normalize_pattern=normalize_hopfield_space,
            normalize_pattern_affine=normalize_hopfield_space_affine)


        # Initialise remaining auxiliary properties.
        if self.association_core.static_execution:
            self.__scaling = 1.0 if scaling is None else scaling
        else:
            assert self.association_core.head_dim > 0, f'invalid hidden dimension encountered.'
        self.__scaling = (1.0 / sqrt(self.association_core.head_dim)) if scaling is None else scaling


        self.__update_steps_max = update_steps_max
        self.__update_steps_eps = update_steps_eps

        self.X_2_Xmean = nn.Sequential(
            Linear(d_model,d_model),
            # nn.Tanh()
        )
        self.X_2_Xlogvar = nn.Sequential(
            Linear(d_model,d_model),
            # nn.Tanh()
        )

        self.topic_2_X = nn.Sequential(
            Linear(d_model,d_model),
            # nn.Tanh()
        )
        self.topic_2_X_mu = nn.Sequential(
            Linear(d_model,d_model),
            # nn.Tanh()
        )
        self.topic_2_X_logvar = nn.Sequential(
            Linear(d_model,d_model),
            # nn.Tanh()
        )

        self.topic_2_X_recon = nn.Sequential(
            Linear(d_model,d_model),
            # nn.Tanh()
        )

        self.X2Q = nn.Sequential(
            Linear(d_model,d_model),
            # nn.Tanh()
        )
        self.X2K = nn.Sequential(
            Linear(d_model,d_model),
            # nn.Tanh()
        )
        self.X2V = nn.Sequential(
            Linear(d_model,d_model),
            # nn.Tanh()
        )
        self.X_all_attn = nn.Sequential(
            Linear(d_model,d_model,bias=False),
            nn.Tanh()
        )
        self.X_all_attn2 = nn.Sequential(
            Linear(d_model,1,bias=False),
        )

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(EnergyTransformerEncoderLayer, self).__setstate__(state)

    def VAE_sample_z(self, mu, log_var):

        eps = Variable(torch.FloatTensor(np.random.randn(*list(mu.size())))).to(args['device'])
        res = mu + torch.exp(log_var / 2) * eps
        return res

    def All_Hopfield_attn(self, X, attn_mask=None, dropout_p = 0.1, training = True, eps = 1e-6):
        # self.W = self.W.to(args['device'])
        # M1 = torch.einsum('bse,ed->bsd',X,self.W)
        # M2 = torch.einsum('bsd,btd->bst',M1,X)  # batch seq seq
        #
        #
        #
        # M3 = self.all_attn_linear(X) # batch seq 1
        # attn_output_weights = M2 + M3.transpose(1,2)

        attn_output, _, attn_output_weights, _ = self.association_core(
            query=X.transpose(0,1), key=X.transpose(0,1), value=X.transpose(0,1),
            key_padding_mask=None, need_weights=True, attn_mask=attn_mask,
            scaling=self.__scaling, update_steps_max=self.__update_steps_max, update_steps_eps=self.__update_steps_eps,
            return_raw_associations=True, return_pattern_projections=False)
        attn_output_weights = attn_output_weights[:,0,:,:]
        self_attn = torch.einsum('bse,bte->bst',X,X)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # attn_output_weights.masked_fill_(attn_mask, float("-inf"))
                self_attn.masked_fill_(attn_mask, float("-inf"))
            else:
                # attn_output_weights += attn_mask
                self_attn += attn_mask

        # attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        # print(self_attn)
        self_attn = F.softmax(self_attn,dim=-1)
        # print(self_attn)

        # print(attn_output_weights,self_attn)

        klq = attn_output_weights[:-1,:] / (self_attn[1:,:]+eps)
        KL = (attn_output_weights[:-1,:] * torch.log(klq + eps))
        # # print(klq, KL)
        KL = KL.sum(2).sum(1).mean(0)

        # KL= (attn_output_weights[:-1,:] - self_attn[1:,:])**2
        # KL = KL.sum(2).sum(1).mean(0)



        # attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
        #
        # attn_output = torch.bmm(attn_output_weights, X)
        # attn_output = self.linear_reweight(attn_output)


        return attn_output.transpose(0,1), KL

    def latent_attn(self, X, attn_mask=None, sentence_mask = None, dropout_p = 0.1, training = True, eps = 1e-6):

        X_mu = self.X_2_Xmean(X)
        X_logvar = self.X_2_Xlogvar(X)

        X_topic = self.VAE_sample_z(X_mu, X_logvar)

        if not training:
            # X_topic = X
            X_topic = X_mu

        X_prime = self.topic_2_X(X_topic)
        recon = ((X-X_prime)*sentence_mask.unsqueeze(2)) **2
        recon = recon.sum(2).mean()

        KL_loss1 = ((0.5 * (torch.exp(X_logvar) + X_mu ** 2 - 1 - X_logvar)) * sentence_mask.unsqueeze(2)) ** 2
        KL_loss1 = KL_loss1.sum(2).mean()

        src = X_topic

        src2 = self.self_attn(src.transpose(0, 1), X.transpose(0, 1), X.transpose(0, 1), attn_mask=attn_mask,
                              key_padding_mask=None)[0]
        src2 = src2.transpose(0, 1)

        src3_mu = self.topic_2_X_mu(src2)
        src3_logvar = self.topic_2_X_logvar(src2)
        src3 = self.VAE_sample_z(src3_mu, src3_logvar)

        if not training:
            src3 = src3_mu
            # src3 = src2

        src3_prime = self.topic_2_X_recon(src3)
        recon1 = ((src3-src3_prime)*sentence_mask.unsqueeze(2)) **2
        recon1 = recon1.sum(2).mean()

        KL_loss2 = ((0.5 * (torch.exp(src3_logvar) + src3_mu ** 2 - 1 - src3_logvar)) * sentence_mask.unsqueeze(2)) ** 2
        KL_loss2 = KL_loss2.sum(2).mean()

        return src3, torch.Tensor([recon + recon1, KL_loss2+ KL_loss1] )


    def All_attn(self, X, attn_mask=None, dropout_p = 0.1, training = True, eps = 1e-6):

        q = self.X2Q(X)
        k = self.X2K(X)
        v = self.X2V(X)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2)) # b s s
        # attn1 = self.X_all_attn(X) # b s e
        # attn_output_weights2 = self.X_all_attn2(attn1)
        # attn_output_weights += attn_output_weights2

        # I = torch.eye(X.size()[1]).to(args['device'])
        # I[0,0] = 0
        # I = I.masked_fill_(I==1, float("-inf"))
        # attn_output_weights = - attn_output_weights

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask
        # attn_output_weights += I

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

        attn_output = torch.bmm(attn_output_weights, v)


        attn_output = self.linear_reweight(attn_output)


        KL = torch.Tensor([0,0]).to(args['device'])

        return attn_output, KL

    def predictcode_attn(self, X, attn_mask=None, dropout_p = 0.1, training = True, eps = 1e-6):
        '''
        :param X: batch seq emb
        :return:
        '''

        src = X

        src2, attn_output_weights = self.self_attn(src.transpose(0, 1), src.transpose(0, 1), src.transpose(0, 1), attn_mask=attn_mask,
                              key_padding_mask=None)
        src2 = src2.transpose(0, 1)
        # attn_output_weights = - torch.bmm(M_inv, raw_attn)

        attn_output_weights_guide = attn_output_weights.detach()
        attn_output_weights_guide = torch.cat([attn_output_weights_guide[1:,:], attn_output_weights_guide[-1,:].unsqueeze(0)], dim = 0)

        attn_mask_01 = (attn_mask == 0)

        attn_output_weights_guide *= attn_mask_01
        # print(attn_output_weights_guide,attn_output_weights_guide.sum(dim=-1,keepdim=True))
        attn_output_weights_guide = attn_output_weights_guide / (attn_output_weights_guide.sum(dim=-1,keepdim=True)+eps)
        # attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
        # print(attn_output_weights, attn_output_weights_guide)
        # attn_output = torch.bmm(attn_output_weights, X)
        # attn_output = self.linear_reweight(attn_output)

        klq = attn_output_weights / (attn_output_weights_guide+eps)
        # print(klq)
        KL = (attn_output_weights * torch.log(klq + eps))
        # print('KL', KL)
        # # print(klq, KL)
        KL = KL.sum(2).sum(1).mean(0)
        return src2, torch.Tensor([0, KL] )

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, training = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # src2 = self.self_attn(src.transpose(0,1), src.transpose(0,1), src.transpose(0,1), attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        # src2 = src2.transpose(0,1)
        # src2 = self.reweight_attn(src, attn_mask=src_mask)
        src2, loss_tuple = self.All_attn(src, attn_mask=src_mask)

        # src2, loss_tuple = self.predictcode_attn(src, attn_mask=src_mask)
        # KL = torch.tensor([0]).to(args['device'])
        # KL = (src2[:, :-1, :] - src[:, 1:, :]) ** 2
        # KL = KL.sum(2).sum(1).mean(0)
        # src2 = torch.Tensor(src.cpu().data).to(args['device'])
        # src2[:,:-1,:] = src[:,1:,:]
        # src_1 = src[:, :-1,:]
        # src2, loss_tuple = self.latent_attn(src, attn_mask=src_mask, sentence_mask=src_key_padding_mask, training=training)
        src = src + self.dropout1(src2)#.transpose(0,1))
        # src[:,1:,:] += src_1
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, loss_tuple



class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt



def _get_clones(module, N):
    if N == 1:
        return ModuleList([module])
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
