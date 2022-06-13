import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..captioning_model import CaptioningModel
from models.beam_search import *
from models.transformer import encoders
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from thop import profile
class Transformer(CaptioningModel):
    def __init__(self, bos_idx, backbone, vlad_encoder, encoder , decoder):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.backbone = backbone
        self.vlad_encoder = vlad_encoder
        self.att_encoder = encoder
        self.decoder = decoder
        # self.random_para = None
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and not 'backbone' in name:
                nn.init.xavier_uniform_(p)

    def forward(self, mode, images, seq=None, max_len=None, eos_idx=None, beam_size=None, out_size=1, return_probs=False):
        if mode == 'xe':
            # with torch.no_grad():
            enc_output, mask_enc = self.backbone(images)
            # features = self.backbone(images)
            # enc_output, mask_enc = self.encoder(features) #[bsz, 49, 512], [bsz,1,1,49]
            enc_output ,mask_enc= self.vlad_encoder(enc_output)
            enc_output ,mask_enc= self.att_encoder(enc_output)
            # dec_output = self.decoder(seq, vlad_output, mask_enc)
            dec_output = self.decoder(seq, enc_output, mask_enc)
            return dec_output
        elif mode == 'rl':
            bs = BeamSearch(self, max_len, eos_idx, beam_size)
            return bs.apply(images, out_size, return_probs)

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]
        
    def print_model_parm_nums(model):
        total = sum([param.nelement() for param in model.parameters()])
        print('  + Number of params: %.2fM' % (total / 1e6))
        class myNet(torch.nn.Module):
            def __init__(self,backbone):
                super(myNet, self).__init__()
                self.net = backbone
            def forward(self,x):
                with torch.no_grad():
                    x = self.net(x)
                return x
            
    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        
        end_backbone = torch.cuda.Event(enable_timing=True)
        end_encoder = torch.cuda.Event(enable_timing=True)
        end_attention = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                # features = self.backbone(visual)
                # self.enc_output, self.mask_enc = self.encoder(features)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
                tensor = torch.randn(1,3,224,224).to(torch.device('cuda'))

                # print(FlopCountAnalysis(self.backbone,tensor).total())
                start.record()
                self.enc_output, self.mask_enc = self.backbone(visual)
                
                end_backbone.record()
                torch.cuda.synchronize()
                self.enc_output, self.mask_enc= self.vlad_encoder(self.enc_output)
                
                end_encoder.record()
                torch.cuda.synchronize()
                # 将原来的concat feature变成att的feature
                self.enc_output, self.mask_enc= self.att_encoder(self.enc_output)
                
                end_attention.record()
                torch.cuda.synchronize()
                
                
                dec_output = self.decoder(it, self.enc_output, self.mask_enc)
                end.record()
                torch.cuda.synchronize()
                
                print("All time = %f" % start.elapsed_time(end))
                print("Decoder time = %f" % (start.elapsed_time(end)-start.elapsed_time(end_attention)))
                print("Attention time = %f" % (start.elapsed_time(end_attention)-start.elapsed_time(end_encoder)))
                print("Vlad time = %f"%(start.elapsed_time(end_encoder)-start.elapsed_time(end_backbone)))
                print("Backbone time = %f" %start.elapsed_time(end_backbone))
            else:
                it = prev_output

        dec_output = self.decoder(it, self.enc_output, self.mask_enc)
        
        return dec_output,torch.zeros(20,49)

        return self.decoder(it, self.enc_output, self.mask_enc), torch.zeros(20,49)
        # return self.decoder(it, self.enc_output, self.mask_enc),self.vis_mat
    

class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            state_dict = {}
            for k, v in state_dict_i.items():
                state_dict[k.split('module.')[-1]] = v
            if 'random_para' in state_dict.keys():
                del state_dict['random_para']
            self.models[i].load_state_dict(state_dict)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        attn_ensemble = []
        for i in range(self.n):
            out_i, att_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))
            attn_ensemble.append(att_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0), torch.mean(torch.cat(attn_ensemble, 0), dim=0)
