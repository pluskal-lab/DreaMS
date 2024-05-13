# WIP

# class Encoder(nn.Module):
#     """Self-attention Encoder"""
#     def __init__(self, args):
#         super(Encoder, self).__init__()
#         self.dropout = args.dropout
#         self.num_layers = args.num_enc_layers
#         self.pre_act = args.pre_act
#
#         self.atts = nn.ModuleList([MultiheadAttention(args) for _ in range(self.num_layers)])
#         self.ffs = nn.ModuleList([FeedForward(args) for _ in range(self.num_layers)])
#
#         num_scales = self.num_layers * 2 + 1 if self.pre_act else self.num_layers * 2
#         if args.scnorm:
#             self.scales = nn.ModuleList([ScaleNorm(args.embed_dim ** 0.5) for _ in range(num_scales)])
#         else:
#             self.scales = nn.ModuleList([nn.LayerNorm(args.embed_dim) for _ in range(num_scales)])
#
#     def forward(self, src_inputs, src_mask):
#         pre_act = self.pre_act
#         post_act = not pre_act
#
#         x = F.dropout(src_inputs, p=self.dropout, training=self.training)
#         for i in range(self.num_layers):
#             att = self.atts[i]
#             ff = self.ffs[i]
#             att_scale = self.scales[2 * i]
#             ff_scale = self.scales[2 * i + 1]
#
#             residual = x
#             x = att_scale(x) if pre_act else x
#             x, _ = att(q=x, k=x, v=x, mask=src_mask)
#             x = residual + F.dropout(x, p=self.dropout, training=self.training)
#             x = att_scale(x) if post_act else x
#
#             residual = x
#             x = ff_scale(x) if pre_act else x
#             x = ff(x)
#             x = residual + F.dropout(x, p=self.dropout, training=self.training)
#             x = ff_scale(x) if post_act else x
#
#         x = self.scales[-1](x) if pre_act else x
#         return x