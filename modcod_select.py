import numpy as np

# MODCOD table
CN0_table = np.array(
    [
        73.85, 74.96, 75.90, 77.20, 78.43, 79.30, 80.23, 80.88, 81.38, 81.70, 82.40, 82.62, 
        82.82, 84.11, 85.17, 85.55, 86.41, 86.89, 87.18, 87.23, 87.81, 88.93, 89.09, 89.33, 
        89.84, 90.48, 91.89, 92.25
    ]
)
SE_table = np.array(
    [
        0.490243, 0.656448, 0.789412, 0.988858, 1.188304, 1.322253, 1.487473, 1.587196, 1.654663, 1.779910, 
        1.766451, 1.788612, 1.980636, 2.228124, 2.637201, 2.478562, 2.966728, 2.646012, 2.679207, 3.165623, 
        3.300184, 3.703295, 3.523143, 3.567342, 3.951571, 4.119540, 4.397854, 4.453027
    ]
)
'''
EbN0_table = np.array([
    -5.36,-4.25,-3.31,-2.01,-0.78,0.09,1.02,1.67,2.17,0.729,3.19,3.41,1.849,3.139,2.949,4.579,4.189,5.919,6.209,5.009,
    5.589,5.740,6.869,7.109,6.65,7.29,8.7,9.06
])
'''
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
sorted_indices = np.argsort(SE_table)
SE_table = SE_table[sorted_indices]
CN0_table = CN0_table[sorted_indices]
#EbN0_table = EbN0_table[sorted_indices]
MOD_table = MOD_table[sorted_indices]
COD_table = COD_table[sorted_indices]


def modcod_select(CN0, link_margin):
    CN0 = CN0 - link_margin # dB
    if CN0 < np.min(CN0_table):
        MOD = "None"
        COD = "None"
        SE = None
    else: 
        # find the indices where the CN0 is greater than the threshold
        indices = np.where(CN0_table <= CN0)[0]
        # find the index of the maximum SE value
        max_index = indices[np.argmax(SE_table[indices])]
        # get the corresponding MOD and COD values
        MOD = MOD_table[max_index]
        COD = COD_table[max_index]
        SE = SE_table[max_index]
    return MOD, COD, SE

