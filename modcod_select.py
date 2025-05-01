import numpy as np

# MODCOD table
EbN0_table = np.array([
    -5.36,-4.25,-3.31,-2.01,-0.78,0.09,1.02,1.67,2.17,0.729,3.19,3.41,1.849,3.139,2.949,4.579,4.189,5.919,6.209,5.009,
    5.589,5.740,6.869,7.109,6.65,7.29,8.7,9.06
])
MOD_table = np.array([
    "QPSK","QPSK","QPSK","QPSK","QPSK","QPSK","QPSK","QPSK","QPSK","8PSK","QPSK","QPSK","8PSK","8PSK","16APSK","8PSK",
    "16APSK","8PSK","8PSK","16APSK","16APSK","32APSK","16APSK",
    "16APSK","32APSK","32APSK","32APSK","32APSK"
])
COD_table = np.array([
    "1/4","1/3","2/5","1/2",
    "3/5","2/3","3/4","4/5","5/6","3/5","8/9","9/10","2/3","3/4","2/3","5/6","3/4","8/9",
    "9/10","4/5","5/6","3/4","8/9","9/10","4/5","5/6","8/9","9/10"
])
sorted_indices = np.argsort(EbN0_table)
EbN0_table = EbN0_table[sorted_indices]
MOD_table = MOD_table[sorted_indices]
COD_table = COD_table[sorted_indices]


def modcod_select(EbN0):
    # given the Eb/N0, find the MODCOD
    if EbN0 < EbN0_table[0]:
        MOD = "None"
        COD = "None"
    elif EbN0 > EbN0_table[-1]:
        MOD = MOD_table[-1]
        COD = COD_table[-1]
    else:
        MOD = MOD_table[np.where(EbN0_table >= EbN0)[0][0]-1]
        COD = COD_table[np.where(EbN0_table >= EbN0)[0][0]-1]
    return MOD, COD

