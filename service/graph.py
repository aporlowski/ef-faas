import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

# Scarp benchmark script_output from script outputs

script_output_dir = os.path.join(os.getcwd(), 'script_output/')
benchmark_output_dir = os.path.join(os.getcwd(), 'benchmark_output/')
print(f'Scraping benchmarks from script script_output at {script_output_dir}')
script_outputs = os.listdir(script_output_dir)
for file in script_outputs:
    with open(f"{script_output_dir}{file}",'r') as f:
        b = open(f"{benchmark_output_dir}{file}-benchmark", "w")
        b.write("csv,timer,status,time,sum,start,tag,uname.node,user,uname.system,platform.version\n")
        for line in f.readlines():
            if line[0:10] == "# csv,test" or line[0:10] == "# csv,main":
                b.write(line[2:]) #keep csv,...
        b.close()

# Read benchmark script_output into df

print(f'Reading benchmarks from benchmark script_output at {benchmark_output_dir}')
columns = ["csv","timer","status","time","sum","start","tag","uname.node","user","uname.system",
           "platform.version", "size", "party", "type"]
benchmark_df = pd.DataFrame(columns=columns)
benchmark_outputs = os.listdir(benchmark_output_dir)
for file in benchmark_outputs:
    df = pd.read_csv(f"{benchmark_output_dir}{file}")
    type = file.split("-")[0]
    party = file.split("-")[2]
    size = file.split("-")[3]
    df['size'] = size
    df['party'] = party
    df['type'] = type
    benchmark_df = pd.concat([benchmark_df, df])

benchmark_df['timer'] = benchmark_df['timer'].str.replace("test/test_", "")
benchmark_df['timer'] = benchmark_df['timer'].str.replace("main/eigenfaces_", "")
benchmark_df['timer'] = benchmark_df['timer'].str.replace("_http", "")
benchmark_df['timer'] = benchmark_df['timer'].str.replace("4", "")
benchmark_df['timer'] = benchmark_df['timer'].str.replace("2", "")

# Compute Statistics
stats_df = benchmark_df.groupby(['size', 'party', 'type', 'timer'], as_index=False)['time'].mean().round(decimals=2)
stats_df.rename(columns={'time' : 'mean', 'timer' : 'test'},inplace=True)
stats_df['min'] = benchmark_df.groupby(['size', 'party', 'type', 'timer'], as_index=False)['time'].min().round(decimals=2)['time']
stats_df['max'] = benchmark_df.groupby(['size', 'party', 'type', 'timer'], as_index=False)['time'].max().round(decimals=2)['time']
stats_df['std'] = benchmark_df.groupby(['size', 'party', 'type', 'timer'], as_index=False)['time'].std().round(decimals=2)['time']
#stats_df['std'] = stats_df['std'] / 5.477225575
#stats_df['std'] = stats_df['std'].round(decimals=2)

#sorter = ['aws', 'azure', 'google', 'mac book', 'docker', 'pi 4', 'pi 3b+']
#stats_df.cloud = stats_df.cloud.astype("category")
#stats_df.cloud.cat.set_categories(sorter, inplace=True)
#stats_df = stats_df.sort_values(["cloud"])

stats_df = stats_df.sort_values(by=['size', 'party', 'type', 'test'])
#----------------------GRAPH Set 1----------------------------

plt.style.use('seaborn-whitegrid')

train_df = stats_df.loc[(stats_df['test'] == 'train')]
upload_df = stats_df.loc[(stats_df['test'] == 'upload')]
predict_df = stats_df.loc[(stats_df['test'] == 'predict')]

graphs = zip(['Train', 'Upload', 'Predict'],[train_df, upload_df, predict_df])

for test, df in graphs:
    labels = df['size'] + '\n' + df['party'] + '\n' + df['type']
    labels.drop_duplicates(inplace=True)
    means = df['mean']
    stds = df['std']
    ind = np.arange(len(labels))
    width = 0.35
    plt.bar(ind, means, width, yerr=stds, capsize=3)
    plt.xlabel("Conditions")
    plt.ylabel("Seconds")
    plt.title(f"{test} Time")
    plt.xticks(ind, labels)
    plt.savefig(f'{test}_graph.png')
    plt.show()

#----------------------GRAPH Set 2----------------------------
pi_series = pd.DataFrame([["aws", "server", "", "train", 35.72, 34.91, 46.50, 1.73],
                          ["azure", "server", "", "train", 40.28, 35.30, 47.50, 3.32],
                          ["google", "server", "", "train", 42.04 ,   41.52 ,   45.93 ,  0.71],
                          ["mac book", "server", "", "train", 33.82 ,   33.82 ,   33.82 ,  0.00 ],
                          ["docker", "server", "", "train", 54.72 ,   54.72 ,   54.72 ,  0.00],
                          ["pi 4", "server", "", "train", 88.59 ,   87.83 ,   89.35 ,  0.32],
                          ["pi 3b+", "server", "", "train", 222.61 ,  208.56 ,  233.48 ,  8.40],
                          ["aws", "client", "", "upload", 0.43 ,    0.16 ,   1.13 , 0.21],
                          ["azure", "client", "", "upload",0.32 ,    0.15 ,   0.50 , 0.15 ],
                          ["google", "client", "", "upload", 0.31 ,    0.18 ,   0.73 , 0.18],
                          ["aws", "client", "", "predict",0.40 ,    0.26 ,   0.80 , 0.18 ],
                          ["azure", "client", "", "predict",  0.36 ,    0.24 ,    0.60 ,  0.13],
                          ["google", "client", "", "predict",  0.36 ,    0.27 ,   0.82 ,  0.16],
                          ],columns=stats_df.columns)
stats_df = stats_df.append(pi_series, ignore_index=False)

train_df = stats_df.loc[(stats_df['test'] == 'train') & (stats_df['party'] == 'server')]
upload_df = stats_df.loc[(stats_df['test'] == 'upload') & (stats_df['party'] == 'client')]
predict_df = stats_df.loc[(stats_df['test'] == 'predict') & (stats_df['party'] == 'client')]

graphs = zip(['Train', 'Upload', 'Predict'],[train_df, upload_df, predict_df])

for test, df in graphs:
    labels = df['size'] + '\n' + df['party'] + '\n' + df['type']
    labels.drop_duplicates(inplace=True)
    means = df['mean']
    stds = df['std']
    ind = np.arange(len(labels))
    width = 0.35
    plt.bar(ind, means, width, yerr=stds, capsize=3)
    plt.xlabel("Conditions")
    plt.ylabel("Seconds")
    plt.title(f"{test} Time")
    plt.xticks(ind, labels)
    if test == 'Train':
        fig = plt.gcf()
        fig.set_size_inches(12, 6.5)
    plt.savefig(f'{test}_platforms_graph.png')
    plt.show()


#-----------------------Print Stats---------------------------

print(stats_df.sort_values(by=['test', 'size', 'party', 'type']).to_markdown(index=False))