import os

import numpy as np
import pandas as pd


def process_whole_seq_data(dir, length):
    folder_path = dir + "/concate"
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tsv')]

    df = pd.DataFrame()

    for csv in csv_files:
        data = pd.read_csv(csv, sep='\t')
        data.drop_duplicates(subset=['payload'], keep='first', inplace=True)
        df = pd.concat([df, data], axis=0)

    # sample payload sequence with response
    df = df[df['response'] != "b''"]
    df = df.reset_index(drop=True)
    df = df[df['payload'].str.len() <= length + 3]

    df.to_csv(str(dir) + "/protocol_payload/emqx_whole_len" + str(length) + ".tsv", sep='\t',
              index=False)

    # format data for gan train
    df = df[['payload']]

    for i in range(len(df)):
        df.iloc[i][0] = df.iloc[i][0][2:-1]
        df.iloc[i][0] = " ".join([p for p in df.iloc[i][0]])
    df.to_csv(str(dir) + "/protocol_payload/train/emqx_whole_len" + str(length) + ".txt", sep='\t',
              header=False, index=False)


def process_state_seq_data(file):
    df = pd.read_csv(file, sep='\t', dtype='bytes')
    dfseq = df[['statelist', 'response_detail']]

    # remove duplicate state sequence
    dfseq.drop_duplicates(subset=['statelist'], keep='first', inplace=True)

    # remove state sequence not trigger response
    dfseq.drop_duplicates(subset=['response_detail'], keep='first', inplace=True)

    dfseq = dfseq.reset_index(drop=True)

    # length sampling
    dfseq = dfseq[dfseq['statelist'].str.count(',') <= 20]
    dfseq = dfseq.reset_index(drop=True)

    # statistic analysis of sequence length
    statelength = dfseq['statelist'].str.count(',').tolist()
    print(np.max(statelength))
    print(np.min(statelength))
    print(np.mean(statelength))
    print(np.median(statelength))

    # format sequence
    for i in range(len(dfseq)):
        dfseq = dfseq[['statelist']]
    dfseq['statelist'][i] = dfseq['statelist'][i][1:-1]

    s1 = [p for p in dfseq['statelist'][i].split(',')]

    dfseq['statelist'][i] = "".join(s1)

    # write to csv file
    dfseq.to_csv("seq_payload/emqx_seq_m0.15g0.85_pure.tsv", sep='\t', header=False, index=False)

    return dfseq


def process_payload_seq_data(dir, length):
    folder_path = dir + "/concate"
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tsv')]

    df = pd.DataFrame()

    for csv in csv_files:
        data = pd.read_csv(csv, sep='\t')
        data.drop_duplicates(subset=['payload'], keep='first', inplace=True)
        df = pd.concat([df, data], axis=0)

    # sample payload sequence with response
    df = df[df['response'] != "b''"]
    df = df.reset_index(drop=True)

    # all mqtt packets
    protocoldict = {}
    statelist = ['CONNECT', 'PINGREQ', 'PUBREL', 'PUBACK', 'PUBCOMP', 'PUBREC', 'PUBLISH', 'SUBSCRIBE', 'UNSUBSCRIBE',
                 'DISCONNECT']
    # statelist = ['CONNECT']

    for s in statelist:
        protocoldict[s] = []
        protocoldict[s + "_LENGTH"] = {}

    for i in range(len(df)):
        l1 = df['statelist'][i][1:-1].replace('\'', '').replace(' ', '').split(',')
        l2 = df['payload_detail'][i][1:-1].replace('\'', '').replace(' ', '').split(',')
        for j, k in zip(l1, l2):
            if k not in protocoldict[j]:
                if len(k) <= length:
                    protocoldict[j].append(k)

    for s in statelist:
        p = np.array([len(payload) for payload in protocoldict[s]])
        protocoldict[s + "_LENGTH"]["max"] = np.max(p)
        protocoldict[s + "_LENGTH"]["min"] = np.min(p)
        protocoldict[s + "_LENGTH"]["mean"] = np.mean(p)
        protocoldict[s + "_LENGTH"]["std"] = np.std(p)

    # packet analysis
    for key in statelist:
        print('{0}: {1}'.format(key, str(len(protocoldict[key]))))
        for lenkey in protocoldict[key + "_LENGTH"].keys():
            print('{0}, {1}: {2}'.format(key + "_LENGTH", lenkey, protocoldict[key + "_LENGTH"][lenkey]))

    for s in statelist:
        if s == 'CONNECT' or s == 'PUBLISH':
            packetlist = [" ".join(p) for p in [pay for pay in protocoldict[s]]]
            dfpacket = pd.DataFrame(data=packetlist, columns=['payload'])
            dfpacket.to_csv(str(dir) + "/protocol_payload/emqx_" + str(s) + "_len" + str(length) + ".tsv", sep='\t',
                            header=False, index=False)
            dfpacket.to_csv(str(dir) + "/protocol_payload/train/emqx_" + str(s) + "_len" + str(length) + ".txt",
                            sep='\t',
                            header=False, index=False)

    return protocoldict


# generate sequence payload
'''
folder_path = os.getcwd() + "/concate"
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tsv')]

protocoldict = {}
protocoldict = process_payload_seq_data(os.getcwd(), 40)
'''

# generate whole payload
process_whole_seq_data(os.getcwd(), 50)
