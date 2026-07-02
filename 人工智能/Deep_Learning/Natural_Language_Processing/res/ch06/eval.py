# coding: utf-8
import sys
sys.path.append('..')
from 人工智能.Deep_Learning.Natural_Language_Processing.res.ch06.rnnlm import Rnnlm
from 人工智能.Deep_Learning.Natural_Language_Processing.res.ch06.better_rnnlm import BetterRnnlm
from 人工智能.Deep_Learning.Natural_Language_Processing.res.dataset import ptb
from 人工智能.Deep_Learning.Natural_Language_Processing.res.common.util import eval_perplexity


if __name__ == '__main__':
    model = Rnnlm()
    #model = BetterRnnlm()

    # 读入训练完的超参数
    model.load_params()

    corpus, _, _ = ptb.load_data('test')

    model.reset_state()
    ppl_test = eval_perplexity(model, corpus)
    print('test perplexity: ', ppl_test)
