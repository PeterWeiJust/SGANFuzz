import binascii
import random as rd
import socket

import pandas as pd

from SGAN.seqgan import SeqGAN
model_dict={}

def init_model(name):
    batch_size = 32
    max_length = 30
    generator_embedding = 64
    generator_hidden = 64
    discriminator_embedding = 64
    discriminator_hidden = 64
    discriminator_dropout = 0.0
    fake_path = "data/train/neg.txt"
    generator_lr = 0.00001
    discriminator_lr = 0.000001
    n_sample = 16
    generator_samples = 10000

    real_path = "data/train/emqx_" + name + "_len30.txt"

    if name == 'connect' or name == 'publish':
        max_length = 40
        real_path = "data/train/emqx_" + str(name) + "_len40.txt"
    elif name == 'seq':
        max_length = 20
        real_path = "data/seq/emqx_seq_pure.txt"
        fake_path = "data/seq/emqx_seq_pure_neg.txt"

    model = SeqGAN(batch_size,max_length,generator_embedding,generator_hidden,discriminator_embedding,discriminator_hidden,discriminator_dropout,path_pos=real_path,path_neg=fake_path,g_lr=generator_lr,d_lr=discriminator_lr,n_sample=n_sample,generate_samples=generator_samples)

    model_dict[name] = model

def load_models(name):
    if name == 'seq':
        g_weights = "data/save/seq/seq_pure_generator.pkl"
        d_weights = "data/save/seq/seq_pure_discriminator.hdf5"
    else:
        g_weights = "data/save/protocol/" + name + "/generator.pkl"
        d_weights = "data/save/protocol/" + name + "/discriminator.hdf5"
    model_dict[name].load(g_weights, d_weights)


def str_to_list(cell):
    cell = cell.split(' ')
    cell = ''.join(c for c in cell)
    cell = cell.strip('\n')
    cell = cell.encode()
    return cell


def read_payload_from_file(filename):
    f = open(filename)
    lines = f.readlines()
    payloadlist = []
    for line in lines:
        line = line.strip()
        if '<' not in line:
            paylist = ''.join(p for p in line.split(' '))
            if len(paylist) % 2 == 0:
                payloadlist.append(paylist.encode())
    return payloadlist


def generate_format_payload(name):
    payload = model_dict[name].generate_single_payload()

    if '<' in payload:
        rdstr1 = rd.choice(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'b', 'c', 'd', 'e', 'f'])
        rdstr2 = rd.choice(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'b', 'c', 'd', 'e', 'f'])
        payload = payload.replace('<S>', rdstr1).replace('<UNK>', rdstr2)

    if len(payload) % 2 == 1:
        payload += '0'

    return payload


def SGAN_Fuzz():
    TARGET_ADDR = "127.0.0.1"
    TARGET_PORT = 1883
    fuzz_num = 30000

    REQUEST_UNIQUE_NUM = 0
    response = {}
    payloads = []

    df_generate = pd.DataFrame(data=None,
                               columns=['payload_detail', 'payload', 'response', 'response_detail', 'statelist',
                                        'unique_response'], dtype=bytes)

    init_model('seq')
    load_models('seq')

    for i in range(fuzz_num):
        print(i)
        payload = generate_format_payload('seq').encode()

        print("Sending payload to server: %s", payload)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.25)
        try:
            s.connect((TARGET_ADDR, TARGET_PORT))
            payload = binascii.unhexlify(payload)
            s.send(payload)

        except ConnectionRefusedError:
            print("No connection was found at %s:%d" % (TARGET_ADDR, TARGET_PORT))
            exit(-1)

        recv = b''
        try:
            recv = s.recv(1024)
            print("Response: %s" % binascii.hexlify(recv))
            if binascii.hexlify(recv) not in response.keys():
                response[binascii.hexlify(recv)] = 1
                print("Found new network response (%d found)" % len(response.keys()))
                REQUEST_UNIQUE_NUM += 1
            else:
                response[binascii.hexlify(recv)] += 1

        except socket.timeout:
            print("Timeout while waiting for response")
        except ConnectionResetError:
            print("Connection closed after sending the payload")

        s.close()

        row = pd.Series([binascii.hexlify(payload), payload, recv, binascii.hexlify(recv), [], REQUEST_UNIQUE_NUM],
                        index=df_generate.columns)
        df_generate = df_generate.append(row, ignore_index=True)

    REQUEST_UNIQUE_NUM = len(response.keys())
    print(response.keys())
    print("REQUEST_UNIQUE_NUM: ", REQUEST_UNIQUE_NUM)


if __name__ == "__main__":
    SGAN_Fuzz()
