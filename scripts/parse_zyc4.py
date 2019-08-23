'''
Created Date: Friday, August 23rd 2019, 9:51:32 pm
Author: Yuchen Zhong
Email: yczhong@hku.hk
'''
import re
import os 

import numpy as np 
import matplotlib.pyplot as plt 

training_job_name = "inception3_25.salus.250iter.0.output"
inference_job_name = "inception3eval_1.salus.2000iter.1.output"
log_path = "scripts/templogs/zyc4"
policy_list = [
    "pack", "fair", "preempt", "mix"
]

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

if __name__=="__main__":   
    ## training  
    lines = []
    fig = plt.figure(figsize=(15,15))
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
    plt.legend(lines, policy_list, prop={'size':30})
    plt.title("exp4 - " + training_job_name)
    plt.xlabel("iter")
    plt.ylabel("images/sec")
    # plt.show()
    os.system("mkdir -p scripts/imgs/exp4")
    plt.savefig('scripts/imgs/exp4/' + training_job_name + ".png", dpi=fig.dpi)

    ## inference 
    for policy in policy_list:
        path = os.path.join(log_path, policy)
        with open(os.path.join(path, inference_job_name), "r") as f:
            content = f.read()
        pattern = re.compile(r"\d+\.\d ")
        items = pattern.findall(content)[100:]
        # print(len(results))
        speeds = list(map(lambda str: float(str.strip()), items))
        latency = list(map(lambda x: 1000/x, speeds))
        
        min_v = np.min(latency)
        max_v = np.max(latency)
        avg_v = np.mean(latency)
        sorted_list = sorted(latency)
        tail_v = sorted_list[int(len(sorted_list)*0.99)]

        print("%s %.2f %.2f %.2f %.2f" % (
            policy, min_v, max_v, avg_v, tail_v
        ))