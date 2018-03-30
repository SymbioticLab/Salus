import parse_log as pl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from importlib import reload
reload(pl)


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
    
    steptf = step25.pivot_table(values='timestamp', index=['op','kernel'], columns='type', aggfunc='first')
    return steptf.reset_index().sort_values(by=['task_ready', 'task_start', 'task_done']).reset_index(drop=True)

steptf = load_tf()

#
# Second load log from salus
#
def load_salus(path):
    logs = pl.load_file(path)
    df = pd.DataFrame(l.__dict__ for l in logs)
    df = df[df.type == 'optracing']
    df = df.drop(['entry_type','level','loc', 'thread', 'timestamp', 'type'], axis=1)
    
    # discard 20 warmup steps and first few steps, use the 5th iteration
    step25 = df[df.step == 132].drop(['step'], axis=1)
    
    # discard some internal or async op: _SOURCE, _Recv, _Send
    ignored = ['_Recv', '_Send']
    step25 = step25[~step25.kernel.isin(ignored)]
    step25 = step25[step25.op != '_SOURCE']
    
    # make sure datetime
    for c in ['task_ready', 'task_sched', 'task_start', 'task_done']:
        step25[c] = pd.to_datetime(step25[c])
    
    # drop unneeded columns
    return step25.sort_values(by=['task_ready', 'task_start', 'task_done']).reset_index(drop=True)

stepsalus = load_salus('logs/optracing/salus/1/perf.output')

#
# Third load 2 session alexnet
#
twosess = load_salus('logs/optracing/salus/2/perf.output')

#
# Draw hlines
#
def draw_lines(ax, step, checkpoints, colors=['g', 'y', 'r'], offset=None):
    """
    step is a pd.DataFrame contains a:
        timestamp, op, kernel, task_ready, task_start, task_done
    """
    # add a name column based on op/kernel
    def name(row):
        return '{}[{}]'.format(row['op'], row['kernel'])
    names = step.apply(name, axis=1).values

    # columns as unix timestamp in us
    columns = [step[c].astype(np.int64) // 10**3 for c in checkpoints]
    # with offset subtracted
    offset = np.min([np.min(col) for col in columns])
    
    # 3 columns as unix timestamp in us
    ready = step.task_ready.astype(np.int64) // 10**3
    start = step.task_start.astype(np.int64) // 10**3
    done = step.task_done.astype(np.int64) // 10**3
    # with offset subtracted
    if offset is None:
        offset = np.min([np.min(c) for c in [ready, start, done]])
    columns = [col - offset for col in columns]
    
    for st, ed, c in zip(columns, columns[1:], colors):
        ax.hlines(y=step.index, xmin=st, xmax=ed, color=c)
    
    # put name on yaxis
    ax.set_yticks(step.index)
    ax.set_yticklabels(names)
    return ax, offset

sns.set_style("dark")
# Turn interactive plotting off
plt.ioff()

# Running one
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(40, 70))
draw_lines(axs[0], steptf, ['task_ready', 'task_start', 'task_done'],
           colors=['g', 'r'])
draw_lines(axs[1], stepsalus, ['task_ready', 'task_sched', 'task_start', 'task_done'],
           colors=['g', 'y', 'r'])
axs[0].set_title('alexnet_25 on TF')
axs[1].set_title('alexnet_25 on Salus')
axs[1].set_xlabel('Normalized time (us)')
fig.tight_layout()
fig.savefig('/home/peifeng/desktop/tfsalus.pdf', dpi=300)
plt.close(fig)

# Running two
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(40, 70))
sessA, sessB = twosess.sess.unique()
alexA = twosess[twosess.sess == sessA].reset_index(drop=True)
alexB = twosess[twosess.sess == sessB].reset_index(drop=True)
ax, offset = draw_lines(axs[0], alexA, ['task_ready', 'task_sched', 'task_start', 'task_done'],
                        colors=['g', 'y', 'r'])
draw_lines(axs[1], alexB, ['task_ready', 'task_sched', 'task_start', 'task_done'],
           colors=['g', 'y', 'r'], offset=offset)
axs[0].set_title('alexnet_25 on Salus (Instance A: {})'.format(sessA))
axs[1].set_title('alexnet_25 on Salus (Instance B: {})'.format(sessB))
axs[1].set_xlabel('Normalized time (us)')
fig.tight_layout()
fig.savefig('/home/peifeng/desktop/salusab.pdf', dpi=300)
plt.close(fig)
