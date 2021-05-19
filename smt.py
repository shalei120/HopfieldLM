import os

import argparse
name2sh = {
    'wk2bet': 'energywiki2_6bet',
    'wk2bet-dm': 'energywiki2_6bet-dm',
    'wk2bet-na': 'energywiki2_6bet-na',
    'wk2bet-nadm': 'energywiki2_6bet-nadm',
    'wk103bet': 'energywiki103_6bet',
    'wk103bet-dm': 'energywiki103_6bet-dm',
    'wk103bet-na': 'energywiki103_6bet-na',
    'wk103bet-nadm': 'energywiki103_6bet-nadm',
    'ewbet': 'energyenwiki8_6bet',
    'ewbet-dm': 'energyenwiki8_6bet-dm',
    'ewbet-na': 'energyenwiki8_6bet-na',
    'ewbet-nadm': 'energyenwiki8_6bet-nadm',

}



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--choose', '-c')
    cmdargs = parser.parse_args()

    if cmdargs.choose is None:
        pass
    else:
        choose = cmdargs.choose

    if choose == 'del':

        for taskname, tasksh in name2sh.items():
            os.system('runai delete '+ taskname)


    elif choose == 'submit':
        for taskname, tasksh in name2sh.items():
            os.system('sh submitdgx.sh -n '+ taskname + ' -m '+tasksh)

