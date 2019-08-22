'''
Created Date: Thursday, August 22nd 2019, 2:08:45 pm
Author: Yuchen Zhong
Email: yczhong@hku.hk
'''
import re
import os 

import numpy as np 
import matplotlib.pyplot as plt 

training_job_name = "resnet50_25.salus.500iter.0.output"
inference_job_name = "resnet50eval_1.salus.1000iter.1.output"
log_path = "scripts/templogs/zyc1"
policy_list = [
    "pack", "fair", "preempt", "mix"
]

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

if __name__=="__main__":    
    lines = []
    for policy in policy_list:
        path = os.path.join(log_path, policy)
        with open(os.path.join(path, training_job_name), "r") as f:
            content = f.read()
        pattern = re.compile(r"images/sec: \d+.\d ")
        items = pattern.findall(content)
        # print(len(results))
        speeds = list(map(lambda str: float(str.split()[1].strip()), items))

        min_v, max_v = int(min(speeds)), int(max(speeds))
        plt.yticks(np.arange(min_v, max_v, 5))
        line, = plt.plot(speeds, linewidth=5.0)
        lines.append(line)
    plt.legend(lines, policy_list)
    plt.title("exp1 - " + training_job_name)
    plt.xlabel("iter")
    plt.ylabel("images/sec")
    plt.show()
        