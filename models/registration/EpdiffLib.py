import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import lagomorph as lm
from lagomorph import adjrep 
from lagomorph import deform 


class Epdiff():
    def __init__(self,alpha=2.0,gamma=1.0):
        # alpha=2.0;gamma = 1.0
        fluid_params = [alpha, 0, gamma]; 
        self.metric = lm.FluidMetric(fluid_params)


    def EPDiff_step(self, m0, dt, phiinv, mommask=None):
        m = adjrep.Ad_star(phiinv, m0)
        if mommask is not None:
            m = m * mommask
        v = self.metric.sharp(m)

        # check the device of phiinv and v
        # print(f'phiinv is on {phiinv.device}')
        # print(f'v is on {v.device}')

        return deform.compose_disp_vel(phiinv, v, dt=-dt), m, v
    

    def my_expmap_seq(self, m0, T=1.0, num_steps=10, phiinv=None, mommask=None, checkpoints=False):
        # t1 = default_timer()
        """
        Given an initial momentum (Lie algebra element), compute the exponential
        map.

        What we return is actually only the inverse transformation phi^{-1}
        """
        m_seq=[]; v_seq=[]; u_seq=[]; ui_seq=[]
        d = len(m0.shape)-2
        v0 = self.metric.sharp(m0)
        m_seq.append(m0); v_seq.append(v0)

        if phiinv is None:
            phiinv = torch.zeros_like(m0)
            phi = torch.zeros_like(m0)

        if checkpoints is None or not checkpoints:
            # skip checkpointing
            dt = T/10
            for i in range(num_steps):
                phiinv, m, v = self.EPDiff_step(m0, dt, phiinv, mommask=mommask)
                u_seq.append(phiinv)
                phi = phi + dt*lm.interp(v, phi)
                ui_seq.append(phi)

                if i<(num_steps-1):
                    m_seq.append(m); v_seq.append(v)
        # print("my_expmap: {}".format(default_timer()-t1))
        return u_seq,v_seq,m_seq, ui_seq    #T-S  V  M   S-T
    
    def my_expmap(self, m0, T=1.0, num_steps=10, phiinv=None, mommask=None, checkpoints=False):
        # t1 = default_timer()
        """
        Given an initial momentum (Lie algebra element), compute the exponential
        map.

        What we return is actually only the inverse transformation phi^{-1}
        """
        m_seq=[]; v_seq=[]; u_seq=[]; ui_seq=[]
        d = len(m0.shape)-2
        v0 = self.metric.sharp(m0)
        m_seq.append(m0); v_seq.append(v0)

        if phiinv is None:
            phiinv = torch.zeros_like(m0)
            phi = torch.zeros_like(m0)

        if checkpoints is None or not checkpoints:
            # skip checkpointing
            dt = T/10
            for i in range(num_steps):
                phiinv, m, v = self.EPDiff_step(m0, dt, phiinv, mommask=mommask)
                u_seq.append(phiinv)
                phi = phi + dt*lm.interp(v, phi)
                ui_seq.append(phi)

                if i<(num_steps-1):
                    m_seq.append(m); v_seq.append(v)
        # print("my_expmap: {}".format(default_timer()-t1))
        # u_seq = [phiinv]
        # v_seq = [v]
        # m_seq = [m]
        # ui_seq = [phi]
        return u_seq,v_seq,m_seq, ui_seq    #T-S  V  M   S-T

    def my_expmap_u2phi(self, m0, T=1.0, num_steps=10, phiinv=None, mommask=None, checkpoints=False):
        """
        Given an initial momentum (Lie algebra element), compute the exponential
        map.

        What we return is actually only the inverse transformation phi^{-1}
        """
        m_seq=[]; v_seq=[]; u_seq=[]; ui_seq=[]
        d = len(m0.shape)-2
        v0 = self.metric.sharp(m0)
        m_seq.append(m0); v_seq.append(v0)


        if phiinv is None:
            phiinv = torch.zeros_like(m0)
            phi = torch.zeros_like(m0)

        if checkpoints is None or not checkpoints:
            dt = T/num_steps
            for i in range(num_steps):
                phiinv, m, v = self.EPDiff_step(m0, dt, phiinv, mommask=mommask)
                u_seq.append(phiinv)
                phi = phi + dt*lm.interp(v, phi)
                ui_seq.append(phi)

                if i<(num_steps-1):
                    m_seq.append(m); v_seq.append(v)


        return u_seq, ui_seq, v_seq
    


    def my_expmap_shooting(self, m0, T=1.0, num_steps=10, phiinv=None, mommask=None, checkpoints=False):
        """
        Given an initial momentum (Lie algebra element), compute the exponential
        map.

        What we return is actually only the inverse transformation phi^{-1}
        """
        # m_seq=[]; v_seq=[]; u_seq=[]
        d = len(m0.shape)-2

        if phiinv is None:
            phiinv = torch.zeros_like(m0)
            phi = torch.zeros_like(m0)

        if checkpoints is None or not checkpoints:
            # skip checkpointing
            dt = T/num_steps
            for i in range(num_steps):
                phiinv, m, v = self.EPDiff_step(m0, dt, phiinv, mommask=mommask)
                phi = phi + dt*lm.interp(v, phi)
        
        return phiinv, phi
    

    

    def lagomorph_expmap_shootin(self, m0, T=1.0, num_steps=10, phiinv=None, mommask=None, checkpoints=False):
        """
        Given an initial momentum (Lie algebra element), compute the exponential map.

        What we return is actually only the inverse transformation phi^{-1}
        """
        d = len(m0.shape)-2

        if phiinv is None:
            phiinv = torch.zeros_like(m0)

        if checkpoints is None or not checkpoints:
            # skip checkpointing
            dt = T/num_steps
            for i in range(num_steps):
                phiinv, m, v = self.EPDiff_step(self.metric, m0, dt, phiinv, mommask=mommask)
                
        return phiinv
    
    def my_get_u(self, v_seq=None, m_seq=None, T=1.0, num_steps=10, phiinv=None):
        if v_seq is None:
            if m_seq is None:
                assert 400>900
            v_seq = [self.metric.sharp(m) for m in m_seq]
        
        dt = T/num_steps
        if phiinv is None:
            phiinv = torch.zeros_like(v_seq[0])

        u_seq = [];phiinv_seq=[]
        for i in range(num_steps):
            phiinv = deform.compose_disp_vel(phiinv, v_seq[i], dt=-dt)
            u_seq.append(phiinv)
            # print(torch.max(phiinv))
        # phiinv_seq = [u+deform.identity for u in u_seq]

        return u_seq
    

    def my_get_u2phi(self, v_seq=None, m_seq=None, T=1.0, num_steps=10, phiinv=None):
        # t1 = default_timer()
        if v_seq is None:
            if m_seq is None:
                assert 400>900
            v_seq = [self.metric.sharp(m) for m in m_seq]
        
        dt = T/num_steps
        if phiinv is None:
            phiinv = torch.zeros_like(v_seq[0])
            phi = torch.zeros_like(v_seq[0])

        u_seq = [];phiinv_seq=[];
        ui_seq = [];phi_seq=[]
        for i in range(num_steps):
            phiinv = deform.compose_disp_vel(phiinv, v_seq[i], dt=-dt)
            u_seq.append(phiinv)
            # print(torch.max(phiinv))
            # phiinv_seq = [u+deform.identity(1, 2, 32,32) for u in u_seq]

            phi = phi + dt*lm.interp(v_seq[i], phi)
            ui_seq.append(phi)
            


        return u_seq, ui_seq



    def my_expmap_advect(self, m, T=1.0, num_steps=10, phiinv=None):
        """Compute EPDiff with vector momenta without using the integrated form.

        This is Euler integration of the following ODE:
            d/dt m = - ad_v^* m
        """
        v_seq = []; m_seq=[]
        d = len(m.shape)-2
        v0 = self.metric.sharp(m)
        m_seq.append(m); v_seq.append(v0)


        if phiinv is None:
            phiinv = torch.zeros_like(m)
        dt = T/num_steps
        v = self.metric.sharp(m)
        phiinv = deform.compose_disp_vel(phiinv, v, dt=-dt)
        v_seq.append(v); m_seq.append(m)


        for i in range(num_steps-1):
            m = m - dt*adjrep.ad_star(v, m)
            v = self.metric.sharp(m)
            phiinv = deform.compose_disp_vel(phiinv, v, dt=-dt)
            if i<(num_steps-2):
                v_seq.append(v); m_seq.append(m)
        return phiinv,v_seq,m_seq



