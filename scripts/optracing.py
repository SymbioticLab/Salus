from __future__ import absolute_import, print_function, division
from builtins import input
import parse_log as pl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from importlib import reload
reload(pl)
#%%

def choose(prompt, choices=None, default=0):
    """Prompt the user to make a choice.

    Args:
        prompt: The prompt to show

        choices: Iterable of tuples of choices. Each tuple represents a choice, and is
        in the form (one letter, help, value) or (one letter, help). If value is missing,
        it defaults to the letter.
        The default choices are [('y', 'yes', True), ('n', 'no', False)]

        default: the index of the default choice. Defaults to 0

    Returns:
        the associated value of the choice the user has made.
    """
    # Handle default arguments
    if choices is None:
        choices = [('y', 'yes', True), ('n', 'no', False)]

    # validate arguments
    if not choices:
        raise ValueError('Empty choices')
    if default < 0 or default >= len(choices):
        raise IndexError(f'Default index should be within [0, {len(choices)}), got: {default}')

    def parse_choice(ch):
        if len(ch) == 2:
            return ch[0].lower(), ch[1], ch[0]
        elif len(ch) == 3:
            return ch[0].lower(), ch[1], ch[2]
        else:
            raise ValueError(f'Invalid choice in choices: {tuple}')

    choices = [parse_choice(c) for c in choices]

    # form choices string
    choices_str = '/'.join(ch[0] if idx != default else ch[0].upper()
                           for idx, ch in enumerate(choices))

    prompt = f'{prompt} [{choices_str}]: '
    def_resp = choices[default][0]
    while True:
        resp = input(prompt)
        if not resp:
            resp = def_resp
        resp = resp.lower()

        for ch, _, value in choices:
            if resp == ch:
                return value

        # Invalid input, print help
        print(f'Invalid response: {resp}')
        print('Accepted responses are:')
        for ch, h, _ in choices:
            print(f'{ch} - {h}')


def confirm(prompt, default=False, yes_choice='y', no_choice='n'):
    """Prompt for user's confirmation on some operation.

    Returns:
        True if the user confirmed, False otherwise.
    """
    return choose(prompt, choices=[(yes_choice, 'yes', True), (no_choice, 'no', False)], default=0 if default else 1)


def select_steps(df):
    # count unique numbers
    counts = df.groupby('step').agg({c: 'nunique' for c in ['kernel', 'op']}).reset_index()
    ss = counts.query('step > 10 & kernel > 10 & op > 200')
    
    # so the step list is
    if len(ss) > 1:
        # drop first iteration
        return ss.step.astype(int).tolist()[1:]
    else:
        slist = []
        # nothing we can find programmatically, let the user decide
        for _, s, ker, op in counts.itertuples():
            if confirm('Step {} has {} tasks, with {} kernels, select?'.format(s, op, ker)):
                slist.append(s)
        return slist


def only_step(steps, idx):
    ss = steps.step.sort_values().unique().tolist()
    if idx >= len(ss):
        idx = len(ss) - 1
    return steps[steps.step == ss[idx]]

def only_steps(steps, idxs):
    ss = steps.step.sort_values().unique().tolist()
    return steps[steps.step.isin([ss[idx] for idx in idxs])]


def unify_names(*dfs):
    # make sure names map to same idx
    names = pd.concat([df.name for df in dfs]).unique()
    ndf = pd.DataFrame({'name': names})
    ndf.index.rename('nameid', inplace=True)
    ndf = ndf.reset_index()
    
    res = [df.drop('nameid', axis=1, errors='ignore').merge(ndf) for df in dfs]
    if len(res) == 1:
        return res[0]
    else:
        return res

#
# First load log from tensorflow
#
tf_events = ['task_ready', 'task_start', 'task_done']
def load_tf(path):
    logs = pl.load_file(path)
    df = pd.DataFrame(l.__dict__ for l in logs)
    df = df[df.type.isin(tf_events)].drop(['level','loc', 'entry_type'], axis=1)
    # make sure step is int
    df['step'] = df.step.astype(int)
    
    ss = select_steps(df)
    step25 = df[df.step.isin(ss)]
    
    # discard some internal or async op: _SOURCE, _Recv, _Send
    ignored = ['_Recv', '_Send']
    step25 = step25[~step25.kernel.isin(ignored)]
    step25 = step25[step25.op != '_SOURCE']
    
    steptf = step25.pivot_table(values='timestamp', index=['step', 'op', 'kernel'],
                                columns='type', aggfunc='first').reset_index()
    
    # add a name column
    def name(row):
        return '{}[{}]'.format(row['op'], row['kernel'])
    steptf['name'] = steptf.apply(name, axis=1).values
    
    # reorder
    steptf = steptf[['step', 'name', 'op', 'kernel'] + tf_events]
    
    return steptf.sort_values(by=tf_events).reset_index(drop=True)

#
# Second load log from salus
#
salus_events = [
    'queued',  # entering queue
    'inspected',  # submitted by scheduler
    'prealloced',  # preallocated
    'running',  # start running in thread pool
    'afterDevCtx',  # get devctx
    'afterPrepInput',  # after prepare input
    'afterCompute',  # after op->Compute
    'afterClearInput',  # after clear input
    'afterPropOut',  # after prop output
    'failed',  # failed
    'done',  # finally
]
def load_salus(path, filter_step=True):
    logs = pl.load_file(path)
    df = pd.DataFrame(l.__dict__ for l in logs)
    df = df[df.type == 'optracing_evt']
    df = df.drop(['entry_type','level','loc', 'thread', 'type'], axis=1)
    # make sure step is int
    df['step'] = df.step.astype(int)
    
    if filter_step:
        ss = select_steps(df)
        step25 = df[df.step.isin(ss)]
    else:
        step25 = df
    
    # discard some internal or async op: _SOURCE, _Recv, _Send
    ignored = ['_Recv', '_Send']
    step25 = step25[~step25.kernel.isin(ignored)]
    step25 = step25[step25.op != '_SOURCE']
    
    # discard unneeded event
    step25 = step25[step25.evt != 'scheduled']
    
    # convert evt values to columns
    step = step25.pivot_table(values='timestamp',
                              index=['step', 'op', 'kernel', 'sess'],
                              columns='evt', aggfunc='last').reset_index()
    # add a name column
    def name(row):
        return '{}[{}]'.format(row['op'], row['kernel'])
    step['name'] = step.apply(name, axis=1).values
    
    # reorder
    step = step[['sess', 'step', 'name', 'op', 'kernel'] + salus_events]
    
    # sort
    return step.sort_values(by=salus_events).reset_index(drop=True)

#
# Draw hlines
#
def draw_lines(ax, step, checkpoints, colors=['g', 'y', 'r'], offset=None,
               labels=None, set_y=True, sort=False):
    """
    step is a pd.DataFrame contains a:
        timestamp, op, kernel, task_ready, task_start, task_done
    """
    # sort first
    if sort:
        step = unify_names(step.sort_values(by=checkpoints))
    # columns as unix timestamp in us
    columns = [step[c].astype(np.int64) // 10**3 for c in checkpoints]
    # with offset subtracted
    if offset is None:
        offset = np.min([np.min(col) for col in columns])
    columns = [col - offset for col in columns]
    
    if labels is None:
        labels = [''] * len(colors)
    for st, ed, c, l in zip(columns, columns[1:], colors, labels):
        ax.hlines(y=step.nameid, xmin=st, xmax=ed, color=c, label=l)
    
    # put name on yaxis
    if set_y:
        ax.set_yticks(step.nameid)
        ax.set_yticklabels(step.name)
    return ax, offset

def draw_lines2(ax, step, checkpoints, colors=['g', 'y', 'r'], offset=None,
               labels=None, set_y=True, sort=False):
    """
    step is a pd.DataFrame contains a:
        timestamp, op, kernel, task_ready, task_start, task_done
    """
    # sort first
    if sort:
        step = unify_names(step.sort_values(by=checkpoints))
    # with offset subtracted
    if offset is None:
        offset = step[checkpoints].min().min()
    columns = [step[col] - offset for col in checkpoints]
    
    if labels is None:
        labels = [''] * len(colors)
    for st, ed, c, l in zip(columns, columns[1:], colors, labels):
        ax.hlines(y=step.nameid, xmin=st, xmax=ed, color=c, label=l)
    
    # put name on yaxis
    if set_y:
        ax.set_yticks(step.nameid)
        ax.set_yticklabels(step.name)
    return ax, offset

def draw_tf(ax, df, **kwargs):
    draw_lines(ax, df, tf_events, colors=['g', 'r'], **kwargs)

def draw_salus(ax, df, **kwargs):
    return draw_lines2(ax, df, [e for e in salus_events if e != 'failed'],
           colors=plt.rcParams['axes.prop_cycle'].by_key()['color'],
           labels=['Queuing', 'Prealloc', 'TPWait', 'DevCtx',
                   'PrepInput', 'Compute', 'ClrInput',
                   'PropOut', 'Misc'],
           **kwargs)

sns.set_style("dark")
#plt.ioff()

#%% def main
def main():
#%%
#
# Set paths
#
    model = 'alexnet_25'
    logdir = 'logs/optracing/'
    outputdir = '/home/peifeng/desktop/'
    figsize = (40, 70)
    set_y = True

#%%
#logdir = 'logs/optracing/'
#outputdir = '/home/peifeng/sync/M.S.Study/Research/salus/groupmeeting/20180403/weekend'
#figsize = None
#set_y = False

#%% Load data
    steptf = load_tf(os.path.join(logdir, 'tf/{}.tf.10iter.0.output'.format(model)))
    stepsalus = load_salus(os.path.join(logdir, 'salus/1/perf.output'))
    twosess = load_salus(os.path.join(logdir, 'salus/2/perf.output'))
    
    steptf = unify_names(steptf)
    stepsalus = unify_names(stepsalus)
    twosess = unify_names(twosess)

#%%
#
# Running one
#
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize)
    draw_lines(axs[0], steptf, tf_events, colors=['g', 'r'], set_y=set_y)
    axs[0].set_title('alexnet_25 on TF')
    # load a few iters
    draw_lines(axs[1], stepsalus, salus_events,
               colors=plt.rcParams['axes.prop_cycle'].by_key()['color'],
               labels=['Queuing', 'Prealloc', 'TPWait', 'DevCtx',
                       'PrepInput', 'Compute', 'ClrInput',
                       'PropOut', 'Misc'],
               set_y=set_y)
    
    axs[1].legend()
    axs[1].set_title('alexnet_25 on Salus')
    axs[1].set_xlabel('Normalized time (us)')
    fig.tight_layout()
    fig.savefig(os.path.join(outputdir, 'tfsalus.pdf'), dpi=300)
    plt.close(fig)
#%%
#
# Running one with matching iter
#
#fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize)
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    draw_lines(axs[0], only_step(steptf, 6), tf_events, colors=['g', 'r'], set_y=set_y)
    axs[0].set_title('alexnet_25 on TensorFlow')
    # use second normal iter
    draw_lines(axs[1], only_step(stepsalus, 10), salus_events,
               colors=plt.rcParams['axes.prop_cycle'].by_key()['color'],
               labels=['Queuing', 'Prealloc', 'TPWait', 'DevCtx',
                       'PrepInput', 'Compute', 'ClrInput',
                       'PropOut', 'Misc'],
               set_y=set_y, sort=True)
    
    axs[1].legend(loc='lower right', bbox_to_anchor=(1,1.5), ncol=2)
    axs[1].set_title('alexnet_25 on Salus')
    axs[1].set_xlabel('Normalized time (us)')
    axs[1].set_ylabel('Tasks')
    axs[1].set_xlim(0, 40000)
    fig.tight_layout()
    fig.savefig(os.path.join(outputdir, 'tfsalus-later.pdf'), dpi=300)
    #plt.close(fig)

#%%
#
# Running two
#
    def split_sess(twosess):
        sessA, sessB = twosess.sess.unique()
        alexA = twosess[twosess.sess == sessA]
        alexB = twosess[twosess.sess == sessB]
        
        return alexA, alexB, sessA, sessB
    
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize)
    
    alexA, alexB, sessA, sessB = split_sess(twosess)
    offset=None
    _, offset = draw_lines(axs[0], alexA, salus_events,
               colors=plt.rcParams['axes.prop_cycle'].by_key()['color'],
               labels=['Queuing', 'Prealloc', 'TPWait', 'DevCtx', 'PrepInput',
                       'Compute', 'ClrInput', 'PropOut', 'Misc'],
                       offset=offset, set_y=set_y)
    _, offset = draw_lines(axs[1], alexB, salus_events,
               colors=plt.rcParams['axes.prop_cycle'].by_key()['color'],
               labels=['Queuing', 'Prealloc', 'TPWait', 'DevCtx', 'PrepInput',
                       'Compute', 'ClrInput', 'PropOut', 'Misc'],
                       offset=offset, set_y=set_y)
    
    axs[0].set_title('alexnet_25 on Salus (Instance A: {})'.format(sessA))
    axs[1].set_title('alexnet_25 on Salus (Instance B: {})'.format(sessB))
    axs[1].set_xlabel('Normalized time (us)')
    fig.tight_layout()
    fig.savefig(os.path.join(outputdir, 'salusab.pdf'), dpi=300)
    plt.close(fig)

#%% CDF of task length

    def cdf(X, ax=None, **kws):
        if ax is None:
            _, ax = plt.subplots()
        n = np.arange(1,len(X)+1) / np.float(len(X))
        Xs = np.sort(X)
        ax.step(Xs, n, **kws)
        ax.set_ylim(0, 1)
        return ax
    
    
    import os
    logdir = 'logs/osdi18/cc/exp18'
    model = 'alexnet_25'
    steptf = load_tf(os.path.join(logdir, 'tf/{}.tf.10iter.0.output'.format(model)))
    stepsalus = load_salus(os.path.join(logdir, 'salus/1/perf.output'))
    
    steptf = unify_names(steptf)
    stepsalus = unify_names(stepsalus)
    
    tflength = steptf[tf_events[-1]] - steptf[tf_events[0]]
    saluslength = stepsalus[salus_events[-1]] - stepsalus[salus_events[0]]
    
    tflength = tflength / pd.Timedelta(microseconds=1)
    saluslength = saluslength / pd.Timedelta(microseconds=1)
    
    plt.style.use(['seaborn-paper', 'mypaper'])
    
    fig, ax = plt.subplots()
    cdf(tflength, ax=ax, label='TensorFlow')
    cdf(saluslength, ax=ax, label='Salus')
    
    ax.set_ylabel('CDF')
    ax.set_xlabel('Tasks')
    ax.legend()
    
    fig.set_size_inches(3.45, 1.75, forward=True)
    fig.tight_layout()
    fig.savefig('/tmp/workspace/exp18.pdf', dpi=300)
    #plt.close()
    
#%% TF compute timeline
    
    def tf_compute_timeline(step, ax=None, **kwargs):
        checkpoints = tf_events
        step = step.copy()
        # columns as unix timestamp in us
        columns = [step[c].astype(np.int64) // 10**3 for c in checkpoints]
        # with offset subtracted
        offset = np.min([np.min(col) for col in columns])
        columns = [col - offset for col in columns]
        
        if ax is None:
            _, ax = plt.subplots()
        ax.hlines(y=np.zeros_like(columns[1]), xmin=columns[1], xmax=columns[2], linewidths=100)
        return ax
    tfop = load_tf('/tmp/workspace/card189/tfop/inception3_100.tf.10iter.0.output')
    ax = tf_compute_timeline(tfop)
    ax.set_xlabel('Time (us)')
            
#%% main
if __name__ == '__main__':
    main()
