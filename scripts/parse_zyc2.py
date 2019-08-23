'''
Created Date: Thursday, August 22nd 2019, 8:17:05 pm
Author: Yuchen Zhong
Email: yczhong@hku.hk
'''
import re
import os 

import numpy as np 
import matplotlib.pyplot as plt 

training_job_name = "resnet50_25.salus.500iter.0.output"
inference_job_names = ["resnet50eval_1.salus.2000iter.1.output",
                       "resnet50eval_1.salus.2000iter.2.output" ]
log_path = "scripts/templogs/zyc2"
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
    plt.title("exp2 - " + training_job_name)
    plt.xlabel("iter")
    plt.ylabel("images/sec")
    # plt.show()
    os.system("mkdir -p scripts/imgs/exp2")
    plt.savefig('scripts/imgs/exp2/' + training_job_name + ".png", dpi=fig.dpi)

    ## inference 
    for policy in policy_list:
        path = os.path.join(log_path, policy)
        items = []
        for name in inference_job_names:
            with open(os.path.join(path, inference_job_names[0]), "r") as f:
                content = f.read()
            pattern = re.compile(r"\d+\.\d ")
            items.extend(pattern.findall(content)[100:])
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