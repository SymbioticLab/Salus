#%%
import parse_log as pl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from importlib import reload
reload(pl)
#%%

#
# First load log from tensorflow
#
def load_tf():
    logs = pl.load_file('logs/optracing/tf/alexnet_25.tf.10iter.0.output')
    df = pd.DataFrame(l.__dict__ for l in logs)
    df = df[df.type != 'unknown'].drop(['level','loc', 'entry_type'], axis=1)
    
    # discard 20 warmup steps and first few steps, use the 5th iteration
    step25 = df[df.step == 20 + 6]
    
    # discard some internal or async op: _SOURCE, _Recv, _Send
    ignored = ['_Recv', '_Send']
    step25 = step25[~step25.kernel.isin(ignored)]
    step25 = step25[step25.op != '_SOURCE']
    
    steptf = step25.pivot_table(values='timestamp', index=['op','kernel'],
                                columns='type', aggfunc='first').reset_index()
    
    # add a name column
    def name(row):
        return '{}[{}]'.format(row['op'], row['kernel'])
    steptf['name'] = steptf.apply(name, axis=1).values
    
    return steptf.sort_values(by=['task_ready', 'task_start', 'task_done']).reset_index(drop=True)

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
    'done',  # finally
]
def load_salus(path, stepid=132):
    logs = pl.load_file(path)
    df = pd.DataFrame(l.__dict__ for l in logs)
    df = df[df.type == 'optracing_evt']
    df = df.drop(['entry_type','level','loc', 'thread', 'type'], axis=1)
    
    # discard 20 warmup steps and first few steps, use the 5th iteration
    step25 = df[df.step == stepid].drop(['step'], axis=1)
    
    # discard some internal or async op: _SOURCE, _Recv, _Send
    ignored = ['_Recv', '_Send']
    step25 = step25[~step25.kernel.isin(ignored)]
    step25 = step25[step25.op != '_SOURCE']
    
    # discard unneeded event
    step25 = step25[step25.evt != 'scheduled']
    
    # convert evt values to columns
    step = step25.pivot_table(values='timestamp',
                              index=['op', 'kernel', 'sess'],
                              columns='evt', aggfunc='first').reset_index()
    # add a name column
    def name(row):
        return '{}[{}]'.format(row['op'], row['kernel'])
    step['name'] = step.apply(name, axis=1).values
    
    # reorder
    step = step[['sess', 'name', 'op', 'kernel'] + salus_events]
    
    # sort
    return step.sort_values(by=salus_events).reset_index(drop=True)

#
# Draw hlines
#
def draw_lines(ax, step, checkpoints, colors=['g', 'y', 'r'], offset=None,
               labels=None):
    """
    step is a pd.DataFrame contains a:
        timestamp, op, kernel, task_ready, task_start, task_done
    """
    # columns as unix timestamp in us
    columns = [step[c].astype(np.int64) // 10**3 for c in checkpoints]
    # with offset subtracted
    if offset is None:
        offset = np.min([np.min(col) for col in columns])
    columns = [col - offset for col in columns]
    
    if labels is None:
        labels = [''] * len(colors)
    for st, ed, c, l in zip(columns, columns[1:], colors, labels):
        ax.hlines(y=step.index, xmin=st, xmax=ed, color=c, label=l)
    
    # put name on yaxis
    ax.set_yticks(step.index)
    ax.set_yticklabels(step.name)
    return ax, offset

sns.set_style("dark")
#%%

#
# Running one
#
steptf = load_tf()
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(40, 70))
draw_lines(axs[0], steptf, ['task_ready', 'task_start', 'task_done'],
           colors=['g', 'r'])

# load a few iters
has_legend = False
offset=None
iters = [132, 133, 134, 135, 136, 137, 138, 139, 140]
#iters = [133]
for i in iters:
    stepsalus = load_salus('logs/optracing/salus/1/perf.output', i)
    ax, offset = draw_lines(axs[1], stepsalus, salus_events,
                            colors=plt.rcParams['axes.prop_cycle'].by_key()['color'],
                            labels=['Queuing', 'Prealloc', 'TPWait', 'DevCtx',
                                    'PrepInput', 'Compute', 'ClrInput',
                                    'PropOut', 'Misc'],
                            offset=offset)
    if not has_legend:
        axs[1].legend()
        has_legend = True
axs[0].set_title('alexnet_25 on TF')
axs[1].set_title('alexnet_25 on Salus')
axs[1].set_xlabel('Normalized time (us)')
fig.tight_layout()
fig.savefig('/home/peifeng/desktop/tfsalus.pdf', dpi=300)
plt.close(fig)
#%%

#%%
#
# Running two
#
def split_sess(twosess):
    sessA, sessB = twosess.sess.unique()
    alexA = twosess[twosess.sess == sessA].reset_index(drop=True)
    alexB = twosess[twosess.sess == sessB].reset_index(drop=True)
    
    # make sure names map to same idx
    names = pd.concat([alexA.name, alexB.name]).unique()
    ndf = pd.DataFrame({'name': names}).reset_index()
    alexA = alexA.merge(ndf).sort_values('index').set_index('index')
    alexB = alexB.merge(ndf).sort_values('index').set_index('index')
    return alexA, alexB, sessA, sessB

# load another iter
offset=None
for i in [133, 134, 135, 136, 137, 138, 139, 140]:
    twosess = load_salus('logs/optracing/salus/2/perf.output', i)
    alexA, alexB, sessA, sessB = split_sess(twosess)
    _, offset = draw_lines(axs[0], alexA, salus_events,
               colors=plt.rcParams['axes.prop_cycle'].by_key()['color'],
               labels=['Queuing', 'Prealloc', 'TPWait', 'DevCtx', 'PrepInput',
                       'Compute', 'ClrInput', 'PropOut', 'Misc'], offset=offset)
    _, offset = draw_lines(axs[1], alexB, salus_events,
               colors=plt.rcParams['axes.prop_cycle'].by_key()['color'],
               labels=['Queuing', 'Prealloc', 'TPWait', 'DevCtx', 'PrepInput',
                       'Compute', 'ClrInput', 'PropOut', 'Misc'], offset=offset)

axs[0].set_title('alexnet_25 on Salus (Instance A: {})'.format(sessA))
axs[1].set_title('alexnet_25 on Salus (Instance B: {})'.format(sessB))
axs[1].set_xlabel('Normalized time (us)')
fig.tight_layout()
fig.savefig('/home/peifeng/desktop/salusab.pdf', dpi=300)
plt.close(fig)
#%%
