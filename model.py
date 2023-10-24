import logging

import numpy as np
import torch
import torch.nn as nn
from math import pi

from LineaRE.utils import rule_match
from config import config


class Model(nn.Module):
    def __init__(self, ent_num, rel_num):
        super(Model, self).__init__()
        self.register_buffer("gamma", torch.tensor(config.gamma))
        self.register_buffer("ents", torch.arange(ent_num).unsqueeze(dim=0))
        self.ent_embd = nn.Embedding(ent_num, config.ent_dim)
        self.rel_embd = nn.Embedding(rel_num, config.rel_dim, max_norm=1.0)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embd.weight)
        nn.init.xavier_uniform_(self.rel_embd.weight)

    def get_pos_embd(self, pos_sample):
        h = self.ent_embd(pos_sample[:, 0]).unsqueeze(dim=1)
        r = self.rel_embd(pos_sample[:, 1]).unsqueeze(dim=1)
        t = self.ent_embd(pos_sample[:, 2]).unsqueeze(dim=1)
        return h, r, t

    def get_neg_embd(self, neg_sample):
        return self.ent_embd(neg_sample)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        raise NotImplementedError


class TransE(Model):
    def __init__(self, ent_num, rel_num):
        super(TransE, self).__init__(ent_num, rel_num)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h, r, t = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score = neg_embd + (r - t)
            elif mode == "tail-batch":
                score = (h + r) - neg_embd
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            score = h + r - t
        score = torch.norm(score, p=config.norm_p, dim=-1) - self.gamma
        return score


class TransH(Model):
    def __init__(self, ent_num, rel_num):
        super(TransH, self).__init__(ent_num, rel_num)
        self.wr = nn.Embedding(rel_num, config.rel_dim)
        nn.init.xavier_uniform_(self.wr.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h, r, t, w = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            wr_neg = (w * neg_embd).sum(dim=-1, keepdim=True)
            wr_neg_wr = wr_neg * w
            if mode == "head-batch":
                wr_t = (w * t).sum(dim=-1, keepdim=True)
                wr_t_wr = wr_t * w
                score = (neg_embd - wr_neg_wr) + (r - (t - wr_t_wr))
            elif mode == "tail-batch":
                wr_h = (w * h).sum(dim=-1, keepdim=True)
                wr_h_wr = wr_h * w
                score = ((h - wr_h_wr) + r) - (neg_embd - wr_neg_wr)
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            wr_h = (w * h).sum(dim=-1, keepdim=True)
            wr_h_wr = wr_h * w
            wr_t = (w * t).sum(dim=-1, keepdim=True)
            wr_t_wr = wr_t * w
            score = (h - wr_h_wr) + r - (t - wr_t_wr)
        score = torch.norm(score, p=config.norm_p, dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        h, r, t = super(TransH, self).get_pos_embd(pos_sample)
        w = self.wr(pos_sample[:, 1]).unsqueeze(dim=1)
        return h, r, t, w


class TransR(Model):
    def __init__(self, ent_num, rel_num):
        super(TransR, self).__init__(ent_num, rel_num)
        self.ent_embd = nn.Embedding(ent_num, config.ent_dim, max_norm=1.0)
        self.rel_embd = nn.Embedding(rel_num, config.ent_dim, max_norm=1.0)
        self.mr = nn.Embedding(rel_num, config.ent_dim * config.rel_dim)
        nn.init.xavier_uniform_(self.mr.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h, r, t, m = self.get_pos_embd(pos_sample)
        m = m.view(-1, 1, config.rel_dim, config.ent_dim)
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            mr_neg = torch.matmul(m, neg_embd.unsqueeze(dim=-1)).squeeze(dim=-1)
            if mode == "head-batch":
                mr_t = torch.matmul(m, t.unsqueeze(dim=-1)).squeeze(dim=-1)
                score = mr_neg + (r - mr_t)
            elif mode == "tail-batch":
                mr_h = torch.matmul(m, h.unsqueeze(dim=-1)).squeeze(dim=-1)
                score = (mr_h + r) - mr_neg
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            mr_h = torch.matmul(m, h.unsqueeze(dim=-1)).squeeze(dim=-1)
            mr_t = torch.matmul(m, t.unsqueeze(dim=-1)).squeeze(dim=-1)
            score = mr_h + r - mr_t
        score = torch.norm(score, p=config.norm_p, dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        h, r, t = super(TransR, self).get_pos_embd(pos_sample)
        m = self.mr(pos_sample[:, 1])
        return h, r, t, m


class SimpleTransR(Model):
    def __init__(self, ent_num, rel_num):
        super(SimpleTransR, self).__init__(ent_num, rel_num)
        self.ent_embd = nn.Embedding(ent_num, config.ent_dim, max_norm=1.0)
        self.rel_embd = nn.Embedding(rel_num, config.rel_dim, max_norm=1.0)
        self.mr = nn.Embedding(rel_num, config.rel_dim)
        nn.init.xavier_uniform_(self.ent_embd.weight)
        nn.init.xavier_uniform_(self.rel_embd.weight)
        nn.init.xavier_uniform_(self.mr.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h, r, t, m = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score = m * neg_embd + (r - m * t)
            elif mode == "tail-batch":
                score = (m * h + r) - m * neg_embd
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            score = m * h + r - m * t
        score = torch.norm(score, p=config.norm_p, dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        h, r, t = super(SimpleTransR, self).get_pos_embd(pos_sample)
        m = self.mr(pos_sample[:, 1]).unsqueeze(dim=1)
        return h, r, t, m


class TransD(Model):
    def __init__(self, ent_num, rel_num):
        super(TransD, self).__init__(ent_num, rel_num)
        # self.ent_embd = nn.Embedding(ent_num, config.ent_dim, max_norm=1.0)
        # self.rel_embd = nn.Embedding(rel_num, config.rel_dim, max_norm=1.0)
        self.ent_p = nn.Embedding(ent_num, config.ent_dim)
        self.rel_p = nn.Embedding(rel_num, config.rel_dim)
        # nn.init.xavier_uniform_(self.ent_embd.weight)
        # nn.init.xavier_uniform_(self.rel_embd.weight)
        nn.init.xavier_uniform_(self.ent_p.weight)
        nn.init.xavier_uniform_(self.rel_p.weight)

    @staticmethod
    def ent_p_rel(ent):
        if config.ent_dim == config.rel_dim:
            return ent
        elif config.ent_dim < config.rel_dim:
            cat = torch.zeros(ent.shape[0], ent.shape[1], config.rel_dim - config.ent_dim)
            if config.cuda:
                cat = cat.cuda()
            return torch.cat([ent, cat], dim=-1)
        else:
            return ent[:, :, :config.rel_dim]

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h, r, t, hp, rp, tp = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_embd, np = self.get_neg_embd(neg_sample)
            np_neg = (np * neg_embd).sum(dim=-1, keepdim=True)
            np_neg_rp = np_neg * rp
            np_neg_rp_n = np_neg_rp + TransD.ent_p_rel(neg_embd)
            if mode == "head-batch":
                tp_t = (tp * t).sum(dim=-1, keepdim=True)
                tp_t_rp = tp_t * rp
                tp_t_rp_t = tp_t_rp + TransD.ent_p_rel(t)
                score = np_neg_rp_n + (r - tp_t_rp_t)
            elif mode == "tail-batch":
                hp_h = (hp * h).sum(dim=-1, keepdim=True)
                hp_h_rp = hp_h * rp
                hp_h_rp_h = hp_h_rp + TransD.ent_p_rel(h)
                score = (hp_h_rp_h + r) - np_neg_rp_n
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            hp_h = (hp * h).sum(dim=-1, keepdim=True)
            hp_h_rp = hp_h * rp
            hp_h_rp_h = hp_h_rp + TransD.ent_p_rel(h)
            tp_t = (tp * t).sum(dim=-1, keepdim=True)
            tp_t_rp = tp_t * rp
            tp_t_rp_t = tp_t_rp + TransD.ent_p_rel(t)
            score = hp_h_rp_h + r - tp_t_rp_t
        score = torch.norm(score, p=config.norm_p, dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        h, r, t = super(TransD, self).get_pos_embd(pos_sample)
        hp = self.ent_p(pos_sample[:, 0]).unsqueeze(dim=1)
        rp = self.rel_p(pos_sample[:, 1]).unsqueeze(dim=1)
        tp = self.ent_p(pos_sample[:, 2]).unsqueeze(dim=1)
        return h, r, t, hp, rp, tp

    def get_neg_embd(self, neg_sample):
        return self.ent_embd(neg_sample), self.ent_p(neg_sample)


class TransIJ(Model):
    def __init__(self, ent_num, rel_num):
        super(TransIJ, self).__init__(ent_num, rel_num)
        self.ent_p = nn.Embedding(ent_num, config.ent_dim)
        nn.init.xavier_uniform_(self.ent_p.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h, r, t, hp, tp = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_embd, np = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                h = neg_embd
                hp = np
            elif mode == "tail-batch":
                t = neg_embd
                tp = np
            else:
                raise ValueError("mode %s not supported" % mode)

        tp_h = (tp * h).sum(dim=-1, keepdim=True)
        hp_tp_h = tp_h * hp
        hp_tp_h_h = hp_tp_h + h
        tp_t = (tp * t).sum(dim=-1, keepdim=True)
        hp_tp_t = tp_t * hp
        hp_tp_t_t = hp_tp_t + t
        score = hp_tp_h_h + r - hp_tp_t_t
        score = torch.norm(score, p=config.norm_p, dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        h, r, t = super(TransIJ, self).get_pos_embd(pos_sample)
        hp = self.ent_p(pos_sample[:, 0]).unsqueeze(dim=1)
        tp = self.ent_p(pos_sample[:, 2]).unsqueeze(dim=1)
        return h, r, t, hp, tp

    def get_neg_embd(self, neg_sample):
        return self.ent_embd(neg_sample), self.ent_p(neg_sample)


class STransE(Model):
    def __init__(self, ent_num, rel_num):
        super(STransE, self).__init__(ent_num, rel_num)
        self.mr1 = nn.Embedding(rel_num, config.ent_dim * config.rel_dim)
        self.mr2 = nn.Embedding(rel_num, config.ent_dim * config.rel_dim)
        nn.init.xavier_uniform_(self.mr1.weight)
        nn.init.xavier_uniform_(self.mr2.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h, r, t, m1, m2 = self.get_pos_embd(pos_sample)
        m1 = m1.view(-1, 1, config.rel_dim, config.ent_dim)
        m2 = m2.view(-1, 1, config.rel_dim, config.ent_dim)
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                mr_neg = torch.matmul(m1, neg_embd.unsqueeze(dim=-1)).squeeze(dim=-1)
                mr_t = torch.matmul(m2, t.unsqueeze(dim=-1)).squeeze(dim=-1)
                score = mr_neg + (r - mr_t)
            elif mode == "tail-batch":
                mr_h = torch.matmul(m1, h.unsqueeze(dim=-1)).squeeze(dim=-1)
                mr_neg = torch.matmul(m2, neg_embd.unsqueeze(dim=-1)).squeeze(dim=-1)
                score = (mr_h + r) - mr_neg
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            mr_h = torch.matmul(m1, h.unsqueeze(dim=-1)).squeeze(dim=-1)
            mr_t = torch.matmul(m2, t.unsqueeze(dim=-1)).squeeze(dim=-1)
            score = mr_h + r - mr_t
        score = torch.norm(score, p=config.norm_p, dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        h, r, t = super(STransE, self).get_pos_embd(pos_sample)
        m1 = self.mr1(pos_sample[:, 1]).unsqueeze(dim=1)
        m2 = self.mr2(pos_sample[:, 1]).unsqueeze(dim=1)
        return h, r, t, m1, m2


class LineaRE_rule_matched(nn.Module):
    def __init__(self, num_ents, num_rels):
        super(LineaRE_rule_matched, self).__init__()
        self.register_buffer('gamma', torch.tensor(config.gamma))
        self.register_buffer('ents', torch.arange(num_ents).unsqueeze(dim=0))
        self.ent_embd = nn.Embedding(num_ents, config.dim, max_norm=None if config.multi_gpu else 1.0, sparse=True)
        self.rel_embd = nn.Embedding(num_rels, config.dim, max_norm=None if config.multi_gpu else 1.0, sparse=True)
        self.wrh = nn.Embedding(num_rels, config.dim)
        self.wrt = nn.Embedding(num_rels, config.dim)
        nn.init.xavier_normal_(self.ent_embd.weight)
        nn.init.xavier_normal_(self.rel_embd.weight)
        nn.init.zeros_(self.wrh.weight)
        nn.init.zeros_(self.wrt.weight)
        self.__dropout = nn.Dropout(config.drop_rate)
        self.__softplus = nn.Softplus(beta=config.beta)
        self.__softmax = nn.Softmax(dim=-1)

        self._log_params()

    def forward(self, sample, w_or_fb, ht, neg_ents=None):
        if neg_ents is not None:
            return self._train_by_rule(sample, w_or_fb, ht, neg_ents)
        else:
            return self._test_by_rule(sample, w_or_fb, ht)

    def _train_by_rule(self, sample, weight, ht, neg_ent):
        h, r, t, wh, wt = self._get_pos_embd(sample)
        rule = []
        r_matched = rule_match(h, r, t, rule)
        neg_embd = self.ent_embd(neg_ent)

        score = self.__dropout(wh * h + r_matched - wt * t)
        pos_score = torch.norm(score, p=config.norm_p, dim=-1) - self.gamma
        pos_score = self.__softplus(torch.squeeze(pos_score, dim=-1))

        if ht == 'head-batch':
            score = self.__dropout(wh * neg_embd + (r_matched - wt * t))
        elif ht == 'tail-batch':
            score = self.__dropout((wh * h + r_matched) - wt * neg_embd)
        else:
            raise ValueError(f'mode {ht} not supported')
        neg_score = self.gamma - torch.norm(score, p=config.norm_p, dim=-1)
        neg_prob = self.__softmax(neg_score * config.alpha).detach()
        neg_score = torch.sum(neg_prob * self.__softplus(neg_score), dim=-1)

        pos_loss = weight * pos_score
        neg_loss = weight * neg_score
        ent_reg, rel_reg = self._regularize()

        return ent_reg, rel_reg, pos_loss, neg_loss

    def _test_by_rule(self, sample, filter_bias, ht):
        h, r, t, wh, wt = self._get_pos_embd(sample)
        if ht == 'head-batch':
            score = wh * self.ent_embd.weight + (r - wt * t)
        elif ht == 'tail-batch':
            score = (wh * h + r) - wt * self.ent_embd.weight
        else:
            raise ValueError(f'mode {ht} not supported')
        score = torch.norm(score, p=config.norm_p, dim=-1) + filter_bias
        return torch.argsort(score)

    def _regularize(self):
        ent_reg = torch.norm(self.ent_embd.weight, p=2, dim=-1)
        rel_reg = torch.norm(self.rel_embd.weight, p=2, dim=-1)
        return ent_reg, rel_reg

    def _get_pos_embd(self, pos_sample):
        h = self.ent_embd(pos_sample[:, 0]).unsqueeze(dim=1)
        r = self.rel_embd(pos_sample[:, 1]).unsqueeze(dim=1)
        t = self.ent_embd(pos_sample[:, 2]).unsqueeze(dim=1)
        wh = self.wrh(pos_sample[:, 1]).unsqueeze(dim=1)
        wt = self.wrt(pos_sample[:, 1]).unsqueeze(dim=1)
        return h, r, t, wh, wt

    def _log_params(self):
        logging.info('>>> Model Parameter Configuration:')
        for name, param in self.named_parameters():
            logging.info(f'Parameter {name}: {str(param.size())}, require_grad = {str(param.requires_grad)}')

    @staticmethod
    def train_step(model, optimizer, data):
        model.train()
        optimizer.zero_grad()
        batch, ht = data
        sample, neg_ents, weight = batch
        ent_reg, rel_reg, pos_loss, neg_loss = model(sample, weight, ht, neg_ents)
        weight_sum = torch.sum(weight)
        pos_loss = torch.sum(pos_loss) / weight_sum
        neg_loss = torch.sum(neg_loss) / weight_sum
        loss = (pos_loss + neg_loss) / 2
        loss += torch.cat([ent_reg ** 2, rel_reg ** 2]).mean() * config.regularization
        loss.backward()
        optimizer.step()
        log = {
            'ent_reg': ent_reg.mean().item(),
            'rel_reg': rel_reg.mean().item(),
            'pos_sample_loss': pos_loss.item(),
            'neg_sample_loss': neg_loss.item(),
            'loss': loss.item()
        }
        return log

    @staticmethod
    def test_step(model, test_dataset_list, detail=False):
        def get_result(ranks_):
            return {
                'MR': np.mean(ranks_),
                'MRR': np.mean(np.reciprocal(ranks_)),
                'HITS@1': np.mean(ranks_ <= 1.0),
                'HITS@3': np.mean(ranks_ <= 3.0),
                'HITS@10': np.mean(ranks_ <= 10.0),
            }

        model.eval()
        mode_ents = {'head-batch': 0, 'tail-batch': 2}
        step = 0
        total_step = sum([len(dataset[0]) for dataset in test_dataset_list])
        ranks = []
        mode_rtps = []
        metrics = []
        with torch.no_grad():
            for test_dataset, mode in test_dataset_list:
                rtps = []
                for pos_sample, filter_bias, rel_tp in test_dataset:
                    pos_sample = pos_sample.to(config.device)
                    filter_bias = filter_bias.to(config.device)
                    sort = model(pos_sample, filter_bias, mode)
                    true_ents = pos_sample[:, mode_ents[mode]].unsqueeze(dim=-1)
                    batch_ranks = torch.nonzero(torch.eq(sort, true_ents), as_tuple=False)
                    ranks.append(batch_ranks[:, 1].detach().cpu().numpy())
                    rtps.append(rel_tp)
                    if step % config.test_log_step == 0:
                        logging.info(f'Evaluating the model... ({step:d}/{total_step:d})')
                    step += 1
                mode_rtps.append(rtps)
            ranks = np.concatenate(ranks).astype(np.float32) + 1.0
            result = get_result(ranks)
            if not detail:
                return result
            metrics.append(result)
            mode_ranks = [ranks[:ranks.size // 2], ranks[ranks.size // 2:]]
            for i in range(2):
                mode_ranks_i = mode_ranks[i]
                rtps = np.concatenate(mode_rtps[i])
                for j in range(1, 5):
                    ranks_tp = mode_ranks_i[rtps == j]
                    result = get_result(ranks_tp)
                    metrics.append(result)
        return metrics


class LineaRE(Model):
    def __init__(self, ent_num, rel_num):
        super(LineaRE, self).__init__(ent_num, rel_num)
        self.wrh = nn.Embedding(rel_num, config.rel_dim, max_norm=config.rel_dim, norm_type=1)
        self.wrt = nn.Embedding(rel_num, config.rel_dim, max_norm=config.rel_dim, norm_type=1)
        nn.init.zeros_(self.wrh.weight)
        nn.init.zeros_(self.wrt.weight)
        nn.init.xavier_uniform_(self.rel_embd.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h, r, t, wh, wt = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score = wh * neg_embd + (r - wt * t)
            elif mode == "tail-batch":
                score = (wh * h + r) - wt * neg_embd
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            score = wh * h + r - wt * t
        score = torch.norm(score, p=config.norm_p, dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        h, r, t = super(LineaRE, self).get_pos_embd(pos_sample)
        wh = self.wrh(pos_sample[:, 1]).unsqueeze(dim=1)
        wt = self.wrt(pos_sample[:, 1]).unsqueeze(dim=1)
        return h, r, t, wh, wt


class DistMult(Model):
    def __init__(self, ent_num, rel_num):
        super(DistMult, self).__init__(ent_num, rel_num)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h, r, t = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score = neg_embd * (r * t)
            elif mode == "tail-batch":
                score = (h * r) * neg_embd
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            score = h * r * t
        return torch.sum(score, dim=-1)


class ComplEx(Model):
    def __init__(self, ent_num, rel_num):
        super(ComplEx, self).__init__(ent_num, rel_num)
        self.ent_embd_im = nn.Embedding(ent_num, config.ent_dim)
        self.rel_embd_im = nn.Embedding(rel_num, config.rel_dim)
        nn.init.xavier_uniform_(self.ent_embd_im.weight)
        nn.init.xavier_uniform_(self.rel_embd_im.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h_re, r_re, t_re, h_im, r_im, t_im = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_re, neg_im = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score_re = t_re * r_re + t_im * r_im
                score_im = t_re * r_im - t_im * r_re
                score = neg_re * score_re - neg_im * score_im
            elif mode == "tail-batch":
                score_re = h_re * r_re - h_im * r_im
                score_im = h_re * r_im + h_im * r_re
                score = score_re * neg_re + score_im * neg_im
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            score_re = h_re * r_re - h_im * r_im
            score_im = h_re * r_im + h_im * r_re
            score = score_re * t_re + score_im * t_im
        return torch.sum(score, dim=-1)

    def get_pos_embd(self, pos_sample):
        h_re, r_re, t_re = super(ComplEx, self).get_pos_embd(pos_sample)
        h_im = self.ent_embd_im(pos_sample[:, 0]).unsqueeze(dim=1)
        r_im = self.rel_embd_im(pos_sample[:, 1]).unsqueeze(dim=1)
        t_im = self.ent_embd_im(pos_sample[:, 2]).unsqueeze(dim=1)
        return h_re, r_re, t_re, h_im, r_im, t_im

    def get_neg_embd(self, neg_sample):
        return self.ent_embd(neg_sample), self.ent_embd_im(neg_sample)


class RotatE(Model):
    def __init__(self, ent_num, rel_num):
        super(RotatE, self).__init__(ent_num, rel_num)
        self.ent_embd_im = nn.Embedding(ent_num, config.ent_dim)
        nn.init.xavier_uniform_(self.ent_embd_im.weight)
        nn.init.uniform_(self.rel_embd.weight, a=-pi, b=pi)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h_re, h_im, r, t_re, t_im = self.get_pos_embd(pos_sample)
        rel_re = torch.cos(r)
        rel_im = torch.sin(r)
        if neg_sample is not None:
            neg_embd_re, neg_embd_im = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score_re = t_re * rel_re + t_im * rel_im
                score_im = t_im * rel_re - t_re * rel_im
            elif mode == "tail-batch":
                score_re = h_re * rel_re - h_im * rel_im
                score_im = h_re * rel_im + h_im * rel_re
            else:
                raise ValueError("mode %s not supported" % mode)
            score_re = score_re - neg_embd_re
            score_im = score_im - neg_embd_im
        else:
            score_re = h_re * rel_re - h_im * rel_im
            score_im = h_re * rel_im + h_im * rel_re
            score_re = score_re - t_re
            score_im = score_im - t_im
        score = torch.stack([score_re, score_im]).norm(dim=0)
        score = score.sum(dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        h_re, r, t_re = super(RotatE, self).get_pos_embd(pos_sample)
        h_im = self.ent_embd_im(pos_sample[:, 0]).unsqueeze(1)
        t_im = self.ent_embd_im(pos_sample[:, 2]).unsqueeze(1)
        return h_re, h_im, r, t_re, t_im

    def get_neg_embd(self, neg_sample):
        return self.ent_embd(neg_sample), self.ent_embd_im(neg_sample)
