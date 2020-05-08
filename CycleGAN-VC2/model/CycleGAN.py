r"""
An implementation of Cycle-GAN-VC2
"""

import torch
import torch.nn as nn
import model.Generator as G
import model.Discriminator as D

class CycleGAN(nn.Module):
    def __init__(self, two_step=False):
        super(CycleGAN, self).__init__()
        self.Gst = G.Generator()
        self.Gts = G.Generator()
        self.Ds = D.Discriminator()
        self.Dt = D.Discriminator()

        self.two_step = two_step
        if two_step:
            self.Ds2 = D.Discriminator()
            self.Dt2 = D.Discriminator()

    def forward(self, source, target):
        r"""
        args:
            source: (B, T=128, F)
            target: (B, T=128, F)
        output:
            d: dictionary
                keys: {'idt_i', 'cyc_i', 'Di_i', 'Di_fake_i'} (i = s, t)
                if two_step = True, {'Di2_i', 'Di2_cyc_i'} is added.
        """

        d = {}
        # For Identity-mapping
        idt_s = self.Gts(source) # G_ts(s)
        idt_t = self.Gst(target) # G_st(t)
        d['idt_s'] = idt_s
        d['idt_t'] = idt_t

        # For Cycle-consistency
        cyc_s = self.Gts(self.Gst(source)) # G_ts(G_st(s))
        cyc_t = self.Gst(self.Gts(target)) # G_st(G_ts(t))
        d['cyc_s'] = cyc_s
        d['cyc_t'] = cyc_t

        # For Adversarial
        Ds_s = self.Ds(source)
        Dt_t = self.Dt(target)
        d['Ds_s'] = Ds_s
        d['Dt_t'] = Dt_t
        Ds_fake_s = self.Ds(self.Gts(target))
        Dt_fake_t = self.Dt(self.Gst(source))
        d['Ds_fake_s'] = Ds_fake_s
        d['Dt_fake_t'] = Dt_fake_t

        # For Two Step Adversarial
        if self.two_step:
            Ds2_s = self.Ds2(source)
            Dt2_t = self.Dt2(target)
            d['Ds2_s'] = Ds2_s
            d['Dt2_t'] = Dt2_t
            Ds2_cyc_s = self.Ds2(cyc_s)
            Dt2_cyc_t = self.Dt2(cyc_t)
            d['Ds2_cyc_s'] = Ds2_cyc_s
            d['Dt2_cyc_t'] = Dt2_cyc_t
        
        # # Appendix
        # d['Gst_t'] = self.Gst(source)

        return d

    def convert_st(self, source):
        r"""
        To show a generated image
        (1, T, 40) -> (1, T, 40)
        """
        return self.Gst(source)

    def convert_ts(self, target):
        r"""
        To show a generated image
        (1, T, 40) -> (1, T, 40)
        """
        return self.Gts(target)
