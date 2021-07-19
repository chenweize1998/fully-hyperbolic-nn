import argparse
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from load_data import Data
from LorentzModel import HyboNet
from optim import RiemannianAdam, RiemannianSGD


class Experiment:
    def __init__(self,
                 args,
                 learning_rate=50,
                 dim=40,
                 nneg=50,
                 valid_steps=10,
                 num_epochs=500,
                 batch_size=128,
                 max_norm=0.5,
                 max_grad_norm=0,
                 optimizer='rsgd',
                 cuda=False,
                 early_stop=10,
                 real_neg=False):
        self.args = args
        self.learning_rate = learning_rate
        self.dim = dim
        self.nneg = nneg
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.max_grad_norm = max_grad_norm
        self.optimizer = optimizer
        self.valid_steps = valid_steps
        self.cuda = cuda
        self.early_stop = early_stop
        self.real_neg = real_neg

    def get_data_idxs(self, data):
        """ Return the training triplets
        """
        data_idxs = [
            (self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]],
             self.entity_idxs[data[i][2]]) for i in range(len(data))
        ]
        return data_idxs

    def get_er_vocab(self, data, idxs=[0, 1, 2]):
        """ Return the valid tail entities for (head, relation) pairs
        Can be used to guarantee that negative samples are true negative.
        """
        er_vocab = defaultdict(set)
        for triple in data:
            er_vocab[(triple[idxs[0]], triple[idxs[1]])].add(triple[idxs[2]])
        return er_vocab

    def evaluate(self, model, data, batch=30):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = np.array(self.get_data_idxs(data))
        sr_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        tt = torch.Tensor(np.array(range(len(d.entities)),
                                   dtype=np.int64)).cuda().long().repeat(
                                       batch, 1)

        for i in range(0, len(test_data_idxs), batch):
            data_point = test_data_idxs[i:i + batch]
            e1_idx = torch.tensor(data_point[:, 0])
            r_idx = torch.tensor(data_point[:, 1])
            e2_idx = torch.tensor(data_point[:, 2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()

            predictions_s = model.forward(
                e1_idx, r_idx, tt[:min(batch,
                                       len(test_data_idxs) - i)])
            for j in range(min(batch, len(test_data_idxs) - i)):

                filt = list(sr_vocab[(data_point[j][0], data_point[j][1])])
                target_value = predictions_s[j][e2_idx[j]].item()
                predictions_s[j][filt] = -np.Inf
                predictions_s[j][e1_idx[j]] = -np.Inf
                predictions_s[j][e2_idx[j]] = target_value

                rank = (predictions_s[j] >= target_value).sum().item() - 1
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        return np.mean(hits[9]), np.mean(hits[2]), np.mean(hits[0]), np.mean(
            1. / np.array(ranks))

    def train_and_eval(self):
        # print("Training the %s model..." % self.model)
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {
            d.relations[i]: i
            for i in range(len(d.relations))
        }

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))
        model = HyboNet(d, self.dim, self.args.max_scale, self.args.max_norm,
                        self.args.margin)
        if self.optimizer == 'radam':
            opt = RiemannianAdam(model.parameters(),
                                 lr=self.learning_rate,
                                 stabilize=1)
        elif self.optimizer == 'rsgd':
            opt = RiemannianSGD(model.parameters(),
                                lr=self.learning_rate,
                                stabilize=1)
        else:
            raise ValueError("Wrong optimizer")
        if self.cuda:
            model.cuda()

        train_data_idxs_np = np.array(train_data_idxs)
        train_data_idxs = torch.tensor(np.array(train_data_idxs)).cuda()
        train_order = list(range(len(train_data_idxs)))

        targets = np.zeros((self.batch_size, self.nneg + 1))
        targets[:, 0] = 1
        targets = torch.FloatTensor(targets).cuda()
        max_mrr = 0.0
        max_it = 0
        mrr = 0
        bad_cnt = 0
        print("Starting training...")
        sr_vocab = self.get_er_vocab(self.get_data_idxs(d.data))
        bar = tqdm(range(1, self.num_epochs + 1),
                   desc='Best:%.3f@%d,curr:%.3f,loss:%.3f' %
                   (max_mrr, max_it, 0., 0.),
                   ncols=75)
        best_model = None
        for it in bar:
            model.train()
            losses = []
            np.random.shuffle(train_order)
            for j in range(0, len(train_data_idxs), self.batch_size):
                data_batch = train_data_idxs[train_order[j:j +
                                                         self.batch_size]]
                data_batch_np = train_data_idxs_np[
                    train_order[j:j + self.batch_size]]
                if j + self.batch_size > len(train_data_idxs):
                    continue

                negsamples = np.random.randint(low=0,
                                               high=len(self.entity_idxs),
                                               size=(data_batch.size(0),
                                                     self.nneg),
                                               dtype=np.int32)
                if self.real_neg:
                    # Filter out the false negative samples. 
                    candidate = np.random.randint(low=0,
                                                  high=len(self.entity_idxs),
                                                  size=(data_batch.size(0)),
                                                  dtype=np.int32)
                    p_candidate = 0
                    e1_idx_np = data_batch_np[:, 0]
                    r_idx_np = data_batch_np[:, 1]
                    for index in range(len(negsamples)):
                        filt = sr_vocab[(e1_idx_np[index], r_idx_np[index])]
                        for index_ in range(len(negsamples[index])):
                            while negsamples[index][index_] in filt:
                                negsamples[index][index_] = candidate[
                                    p_candidate]
                                p_candidate += 1
                                if p_candidate == len(candidate):
                                    candidate = np.random.randint(
                                        0,
                                        len(self.entity_idxs),
                                        size=(self.batch_size))
                                    p_candidate = 0
                negsamples = torch.LongTensor(negsamples).cuda()

                opt.zero_grad()

                e1_idx = data_batch[:, 0]
                r_idx = data_batch[:, 1]
                e2_idx = torch.cat([data_batch[:, 2:3], negsamples], dim=-1)

                predictions = model.forward(e1_idx, r_idx, e2_idx)
                loss = model.loss(predictions, targets)
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        model.parameters(), max_norm=self.max_grad_norm)
                opt.step()
                losses.append(loss.item())
            bar.set_description('Best:%.4f@%d,cur:%.4f,loss:%.3f' %
                                (max_mrr, max_it, mrr, np.mean(losses)))
            model.eval()
            with torch.no_grad():
                if not it % self.valid_steps:
                    hit10, hit3, hit1, mrr = self.evaluate(model, d.valid_data)
                    if mrr > max_mrr:
                        max_mrr = mrr
                        max_it = it
                        bad_cnt = 0
                        best_model = deepcopy(model.state_dict())
                    else:
                        bad_cnt += 1
                        if bad_cnt == self.early_stop:
                            break
                    bar.set_description('Best:%.4f@%d,cur:%.4f,loss:%.3f' %
                                        (max_mrr, max_it, mrr, loss.item()))
        with torch.no_grad():
            model.load_state_dict(best_model)
            hit10, hit3, hit1, mrr = self.evaluate(model, d.test_data)
        print(
            'Test Result\nBest it:%d\nHit@10:%f\nHit@3:%f\nHit@1:%f\nMRR:%f' %
            (max_it, hit10, hit3, hit1, mrr))
        # file = 'log_%d.%s.txt' % (args.dim, args.dataset)
        # with open(file, 'a') as f:
        #     f.write(str(args) + '\n')
        #     f.write('Best it:%d\nHit@10:%f\nHit@3:%f\nHit@1:%f\nMRR:%f\n' % (max_it, hit10, hit3, hit1, mrr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str,
                        default="FB15k-237",
                        nargs="?",
                        help="Which dataset to use: FB15k-237 or WN18RR.")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=500,
                        nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        nargs="?",
                        help="Batch size.")
    parser.add_argument("--nneg",
                        type=int,
                        default=50,
                        nargs="?",
                        help="Number of negative samples.")
    parser.add_argument("--lr",
                        type=float,
                        default=5,
                        nargs="?",
                        help="Learning rate.")
    parser.add_argument("--dim",
                        type=int,
                        default=40,
                        nargs="?",
                        help="Embedding dimensionality.")
    parser.add_argument('--early_stop', default=10, type=int)
    parser.add_argument('--max_norm', default=2, type=float)
    parser.add_argument('--max_scale', default=2, type=float)
    parser.add_argument('--margin', default=6, type=float)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--real_neg', action='store_true')
    parser.add_argument('--optimizer',
                        choices=['rsgd', 'radam'],
                        default='rsgd')
    parser.add_argument('--valid_steps', default=10, type=int)
    parser.add_argument("--cuda",
                        type=bool,
                        default=True,
                        nargs="?",
                        help="Whether to use cuda (GPU) or not (CPU).")

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True
    seed = 40
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = Data(data_dir=data_dir)
    print(args)
    experiment = Experiment(args,
                            learning_rate=args.lr,
                            batch_size=args.batch_size,
                            num_epochs=args.num_epochs,
                            dim=args.dim,
                            cuda=args.cuda,
                            nneg=args.nneg,
                            max_norm=args.max_grad_norm,
                            optimizer=args.optimizer,
                            valid_steps=args.valid_steps,
                            max_grad_norm=args.max_grad_norm,
                            early_stop=args.early_stop,
                            real_neg=args.real_neg)
    experiment.train_and_eval()
