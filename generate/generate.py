import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from utils import check_novelty, sample, canonic_smiles
from dataset import SmileDataset
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
import math
from tqdm import tqdm
import argparse
from model import GPT, GPTConfig
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from moses.utils import get_mol
import re
import moses
import json
from rdkit.Chem import RDConfig
import json

import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

import sascorer

from rdkit.Chem.rdMolDescriptors import CalcTPSA

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_weight', type=str, help="path of model weights", required=False, default='only_smalldataset.pt')
        parser.add_argument('--scaffold', action='store_true', default=True, help='condition on scaffold')
        parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
        parser.add_argument('--csv_name', type=str, help="name to save the generated mols in csv format", required=False,default='only_smalldataset')
        parser.add_argument('--data_name', type=str, default='egfr_moses73', help="name of the dataset to train on",
                            required=False)
        parser.add_argument('--batch_size', type=int, default=512, help="batch size", required=False)
        parser.add_argument('--gen_size', type=int, default=1024, help="number of times to generate from a batch",
                            required=False)
        parser.add_argument('--vocab_size', type=int, default=94, help="number of layers",
                            required=False)
        parser.add_argument('--block_size', type=int, default=78, help="number of layers",
                            required=False)
        parser.add_argument('--props', nargs="+", default=[], help="properties to be used for condition", required=False)
        parser.add_argument('--n_layer', type=int, default=8, help="number of layers", required=False)
        parser.add_argument('--n_head', type=int, default=8, help="number of heads", required=False)
        parser.add_argument('--n_embd', type=int, default=256, help="embedding dimension", required=False)
        parser.add_argument('--lstm_layers', type=int, default=2, help="number of layers in lstm", required=False)

        args = parser.parse_args()


        context = "C"


        data = pd.read_csv('../train/datasets/' + args.data_name + '.csv')
        data = data.dropna(axis=0).reset_index(drop=True)
        data.columns = data.columns.str.lower()

        if 'moses' in args.data_name:
            smiles = data[data['split']!='test_scaffolds']['smiles']   # needed for moses
            scaf = data[data['split']!='test_scaffolds']['scaffold_smiles']   # needed for moses

        pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)

        
        if ('moses' in args.data_name) and args.scaffold:
            #scaffold_max_len=48
            scaffold_max_len=68
        else:
            scaffold_max_len = 100

        stoi = json.load(open(f'moses3_stoi.json', 'r'))
        #stoi = json.load(open(f'{args.data_name}_stoi.json', 'r'))

        itos = { i:ch for ch,i in stoi.items() }

        print(itos)
        print(len(itos))

        num_props = len(args.props)
        mconf = GPTConfig(args.vocab_size, args.block_size, num_props = num_props,
                       n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, scaffold = args.scaffold, scaffold_maxlen = scaffold_max_len,
                       lstm = args.lstm, lstm_layers = args.lstm_layers)
        model = GPT(mconf)

        model = torch.load('/home/gcn/code/molgpt/train_trans/test.pt')

        model.to('cuda')
        print('Model loaded')

        gen_iter = math.ceil(args.gen_size / args.batch_size)

        if 'guacamol' in args.data_name:
            prop2value = {'qed': [0.3, 0.5, 0.7], 'sas': [2.0, 3.0, 4.0], 'logp': [2.0, 4.0, 6.0], 'tpsa': [40.0, 80.0, 120.0],
                        'tpsa_logp': [[40.0, 2.0], [80.0, 2.0], [120.0, 2.0], [40.0, 4.0], [80.0, 4.0], [120.0, 4.0], [40.0, 6.0], [80.0, 6.0], [120.0, 6.0]],
                        'sas_logp': [[2.0, 2.0], [2.0, 4.0], [2.0, 6.0], [3.0, 2.0], [3.0, 4.0], [3.0, 6.0], [4.0, 2.0], [4.0, 4.0], [4.0, 6.0]],
                        'tpsa_sas': [[40.0, 2.0], [80.0, 2.0], [120.0, 2.0], [40.0, 3.0], [80.0, 3.0], [120.0, 3.0], [40.0, 4.0], [80.0, 4.0], [120.0, 4.0]],
                        'tpsa_logp_sas': [[40.0, 2.0, 2.0], [40.0, 2.0, 4.0], [40.0, 6.0, 4.0], [40.0, 6.0, 2.0], [80.0, 6.0, 4.0], [80.0, 2.0, 4.0], [80.0, 2.0, 2.0], [80.0, 6.0, 2.0]]}
        else:
            prop2value = {'qed': [0.6, 0.725, 0.85], 'sas': [2.0, 2.75, 3.5], 'logp': [1.0, 2.0, 3.0], 'tpsa': [30.0, 60.0, 90.0],
                        'tpsa_logp': [[40.0, 2.0], [80.0, 2.0], [40.0, 4.0], [80.0, 4.0]],
                        'sas_logp': [[2.0, 1.0], [2.0, 3.0], [3.5, 1.0], [3.5, 3.0]],
                        'tpsa_sas': [[40.0, 2.0], [80.0, 2.0], [40.0, 3.5], [80.0, 3.5]],
                        'tpsa_logp_sas': [[40.0, 1.0, 2.0], [40.0, 1.0, 3.5], [40.0, 3.0, 2.0], [40.0, 3.0, 3.5], [80.0, 1.0, 2.0], [80.0, 1.0, 3.5], [80.0, 3.0, 2.0], [80.0, 3.0, 3.5]]}
            
        
        prop_condition = None
        if len(args.props) > 0:
            prop_condition = prop2value['_'.join(args.props)]
        
        scaf_condition = None

        if args.scaffold:
            scaf_condition = ['O=C(Cc1ccccc1)NCc1ccccc1', 'c1cnc2[nH]ccc2c1', 'c1ccc(-c2ccnnc2)cc1', 'c1ccc(-n2cnc3ccccc32)cc1', 'O=C(c1cc[nH]c1)N1CCN(c2ccccc2)CC1']
            scaf_condition = [ i + str('<')*(scaffold_max_len - len(regex.findall(i))) for i in scaf_condition]


        all_dfs = []
        all_metrics = []
        count = 0
        if prop_condition is None and scaf_condition is None:
            molecules = []
            count += 1
            for i in tqdm(range(gen_iter)):
                    x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
                    p = None
                    sca = None
                    y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None, prop = p, scaffold = sca)   # 0.7 for guacamol
                    for gen_mol in y:
                            completion = ''.join([itos[int(i)] for i in gen_mol])
                            completion = completion.replace('<', '')
                            mol = get_mol(completion)
                            if mol:
                                    molecules.append(mol)

            "Valid molecules % = {}".format(len(molecules))

            mol_dict = []

            for i in molecules:
                    mol_dict.append({'molecule' : i, 'smiles': Chem.MolToSmiles(i)})

            
            results = pd.DataFrame(mol_dict)

            
            canon_smiles = [canonic_smiles(s) for s in results['smiles']]
            unique_smiles = list(set(canon_smiles))
            if 'moses' in args.data_name:
                    novel_ratio = check_novelty(unique_smiles, set(data[data['split']=='train']['smiles']))   # replace 'source' with 'split' for moses
            else:
                    novel_ratio = check_novelty(unique_smiles, set(data[data['source']=='train']['smiles']))   # replace 'source' with 'split' for moses


            print('Valid ratio: ', np.round(len(results)/(args.batch_size*gen_iter), 3))
            print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
            print('Novelty ratio: ', np.round(novel_ratio/100, 3))

            
            results['qed'] = results['molecule'].apply(lambda x: QED.qed(x) )
            results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
            results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x) )
            results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x) )
            results['validity'] = np.round(len(results)/(args.batch_size*gen_iter), 3)
            results['unique'] = np.round(len(unique_smiles)/len(results), 3)
            results['novelty'] = np.round(novel_ratio/100, 3)
            all_dfs.append(results)

        
        elif (prop_condition is not None) and (scaf_condition is None):
            count = 0
            for c in prop_condition:
                molecules = []
                count += 1
                for i in tqdm(range(gen_iter)):
                        x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
                        p = None
                        if len(args.props) == 1:
                                p = torch.tensor([[c]]).repeat(args.batch_size, 1).to('cuda')   # for single condition
                        else:
                                p = torch.tensor([c]).repeat(args.batch_size, 1).unsqueeze(1).to('cuda')    # for multiple conditions
                        sca = None
                        y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None, prop = p, scaffold = sca)   # 0.7 for guacamol
                        for gen_mol in y:
                                completion = ''.join([itos[int(i)] for i in gen_mol])
                                completion = completion.replace('<', '')
                                mol = get_mol(completion)
                                if mol:
                                        molecules.append(mol)

                "Valid molecules % = {}".format(len(molecules))

                mol_dict = []

                for i in molecules:
                        mol_dict.append({'molecule' : i, 'smiles': Chem.MolToSmiles(i)})


                results = pd.DataFrame(mol_dict)


                canon_smiles = [canonic_smiles(s) for s in results['smiles']]
                unique_smiles = list(set(canon_smiles))
                if 'moses' in args.data_name:
                        novel_ratio = check_novelty(unique_smiles, set(data[data['split']=='train']['smiles']))   # replace 'source' with 'split' for moses
                else:
                        novel_ratio = check_novelty(unique_smiles, set(data[data['source']=='train']['smiles']))   # replace 'source' with 'split' for moses


                print(f'Condition: {c}')
                print('Valid ratio: ', np.round(len(results)/(args.batch_size*gen_iter), 3))
                print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
                print('Novelty ratio: ', np.round(novel_ratio/100, 3))

                
                if len(args.props) == 1:
                        results['condition'] = c
                elif len(args.props) == 2:
                        results['condition'] = str((c[0], c[1]))
                else:
                        results['condition'] = str((c[0], c[1], c[2]))
                        
                results['qed'] = results['molecule'].apply(lambda x: QED.qed(x) )
                results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
                results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x) )
                results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x) )
                # results['temperature'] = temp
                results['validity'] = np.round(len(results)/(args.batch_size*gen_iter), 3)
                results['unique'] = np.round(len(unique_smiles)/len(results), 3)
                results['novelty'] = np.round(novel_ratio/100, 3)
                all_dfs.append(results)


        elif prop_condition is None and scaf_condition is not None:
            count = 0
            for j in scaf_condition:
                molecules = []
                count += 1
                for i in tqdm(range(gen_iter)):
                    x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
                    p = None
                    sca = torch.tensor([stoi[s] for s in regex.findall(j)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
                    y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None, prop = p, scaffold = sca)   # 0.7 for guacamol
                    for gen_mol in y:
                            completion = ''.join([itos[int(i)] for i in gen_mol])
                            completion = completion.replace('<', '')
                            mol = get_mol(completion)
                            if mol:
                                    molecules.append(mol)                                


                "Valid molecules % = {}".format(len(molecules))

                mol_dict = []

                for i in molecules:
                        mol_dict.append({'molecule' : i, 'smiles': Chem.MolToSmiles(i)})


                results = pd.DataFrame(mol_dict)


                canon_smiles = [canonic_smiles(s) for s in results['smiles']]
                unique_smiles = list(set(canon_smiles))
                if 'moses' in args.data_name:
                        novel_ratio = check_novelty(unique_smiles, set(data[data['split']=='train']['smiles']))   # replace 'source' with 'split' for moses
                else:
                        novel_ratio = check_novelty(unique_smiles, set(data[data['source']=='train']['smiles']))   # replace 'source' with 'split' for moses


                print(f'Scaffold: {j}')
                print('Valid ratio: ', np.round(len(results)/(args.batch_size*gen_iter), 3))
                print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
                print('Novelty ratio: ', np.round(novel_ratio/100, 3))

                
                        
                results['scaffold_cond'] = j
                results['qed'] = results['molecule'].apply(lambda x: QED.qed(x) )
                results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
                results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x) )
                results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x) )
                results['validity'] = np.round(len(results)/(args.batch_size*gen_iter), 3)
                results['unique'] = np.round(len(unique_smiles)/len(results), 3)
                results['novelty'] = np.round(novel_ratio/100, 3)
                all_dfs.append(results)


        elif prop_condition is not None and scaf_condition is not None:
            count = 0
            for j in scaf_condition:
                for c in prop_condition:
                    molecules = []
                    count += 1
                    for i in tqdm(range(gen_iter)):
                        x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
                        p = None
                        if len(args.props) == 1:
                                p = torch.tensor([[c]]).repeat(args.batch_size, 1).to('cuda')   # for single condition
                        else:
                                p = torch.tensor([c]).repeat(args.batch_size, 1).unsqueeze(1).to('cuda')    # for multiple conditions
                        sca = torch.tensor([stoi[s] for s in regex.findall(j)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')
                        y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None, prop = p, scaffold = sca)   # 0.7 for guacamol
                        for gen_mol in y:
                                completion = ''.join([itos[int(i)] for i in gen_mol])
                                completion = completion.replace('<', '')
                                mol = get_mol(completion)
                                if mol:
                                        molecules.append(mol)                                


                    "Valid molecules % = {}".format(len(molecules))

                    mol_dict = []

                    for i in molecules:
                            mol_dict.append({'molecule' : i, 'smiles': Chem.MolToSmiles(i)})

                    
                    canon_smiles = [canonic_smiles(s) for s in results['smiles']]
                    unique_smiles = list(set(canon_smiles))
                    if 'moses' in args.data_name:
                            novel_ratio = check_novelty(unique_smiles, set(data[data['split']=='train']['smiles']))   # replace 'source' with 'split' for moses
                    else:
                            novel_ratio = check_novelty(unique_smiles, set(data[data['source']=='train']['smiles']))   # replace 'source' with 'split' for moses


                    print(f'Condition: {c}')
                    print(f'Scaffold: {j}')
                    print('Valid ratio: ', np.round(len(results)/(args.batch_size*gen_iter), 3))
                    print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
                    print('Novelty ratio: ', np.round(novel_ratio/100, 3))

                    
                    if len(args.props) == 1:
                            results['condition'] = c
                    elif len(args.props) == 2:
                            results['condition'] = str((c[0], c[1]))
                    else:
                            results['condition'] = str((c[0], c[1], c[2]))
                            
                    results['scaffold_cond'] = j
                    results['qed'] = results['molecule'].apply(lambda x: QED.qed(x) )
                    results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
                    results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x) )
                    results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x) )
                    results['validity'] = np.round(len(results)/(args.batch_size*gen_iter), 3)
                    results['unique'] = np.round(len(unique_smiles)/len(results), 3)
                    results['novelty'] = np.round(novel_ratio/100, 3)
                    all_dfs.append(results)


        results = pd.concat(all_dfs)
        results.to_csv('gen_csv_again/' + args.csv_name + '.csv', index = False)

        unique_smiles = list(set(results['smiles']))
        canon_smiles = [canonic_smiles(s) for s in results['smiles']]
        unique_smiles = list(set(canon_smiles))
        if 'moses' in args.data_name:
                novel_ratio = check_novelty(unique_smiles, set(data[data['split']=='train']['smiles']))    # replace 'source' with 'split' for moses
        else:
                novel_ratio = check_novelty(unique_smiles, set(data[data['source']=='train']['smiles']))    # replace 'source' with 'split' for moses
               

        print('Valid ratio: ', np.round(len(results)/(args.batch_size*gen_iter*count), 3))
        print('Unique ratio: ', np.round(len(unique_smiles)/len(results), 3))
        print('Novelty ratio: ', np.round(novel_ratio/100, 3))

