

import heapq
#import bisect
import numpy as np
import pandas as pd

import statistics
from sklearn.preprocessing import OrdinalEncoder
#from xgboost import XGBRanker

#import shap
import math
import random

# DOT file visualization
# https://dreampuf.github.io/GraphvizOnline/?engine=dot#digraph%20G%20%7B%0A%0A%20%20subgraph%20cluster_0%20%7B%0A%20%20%20%20style%3Dfilled%3B%0A%20%20%20%20color%3Dlightgrey%3B%0A%20%20%20%20node%20%5Bstyle%3Dfilled%2Ccolor%3Dwhite%5D%3B%0A%20%20%20%20a0%20-%3E%20a1%20-%3E%20a2%20-%3E%20a3%3B%0A%20%20%20%20label%20%3D%20%22process%20%231%22%3B%0A%20%20%7D%0A%0A%20%20subgraph%20cluster_1%20%7B%0A%20%20%20%20node%20%5Bstyle%3Dfilled%5D%3B%0A%20%20%20%20b0%20-%3E%20b1%20-%3E%20b2%20-%3E%20b3%3B%0A%20%20%20%20label%20%3D%20%22process%20%232%22%3B%0A%20%20%20%20color%3Dblue%0A%20%20%7D%0A%20%20start%20-%3E%20a0%3B%0A%20%20start%20-%3E%20b0%3B%0A%20%20a1%20-%3E%20b3%3B%0A%20%20b2%20-%3E%20a3%3B%0A%20%20a3%20-%3E%20a0%3B%0A%20%20a3%20-%3E%20end%3B%0A%20%20b3%20-%3E%20end%3B%0A%0A%20%20start%20%5Bshape%3DMdiamond%5D%3B%0A%20%20end%20%5Bshape%3DMsquare%5D%3B%0A%7D

class TreeNode:
    def __init__(self, param_name=None, value=None, score=0, depth=0):
        self.param = param_name
        self.value = value
        self.score = score
        self.depth = depth
        self.children = []
        self.parent = None
        self.mask = False
        
    #def __lt__(self, other):
    #    return -self.score < -other.score


def beam_search(param_order, main_shap, interact_shap, beam_width=10):
    root = TreeNode(value='Root', depth=0)  # Root node (no parameters)
    heap = [(-0, root)]       # Priority queue maintains expansion nodes (sorted by score)
    
    for depth, param in enumerate(param_order, 1):
        new_heap = []
        for _, node in heap:
            # Get all possible values ​​of the current parameter, sorted by main SHAP
            candidates = sorted(main_shap[param].items(), key=lambda x: -x[1])
            for value, value_shap in candidates[:beam_width]:
                # Calculate the current node score
                new_score = node.score + value_shap

                # Adding interact SHAP score
                #for parent in node
                parent = node
                while True:
                    parent = parent.parent
                    if parent is None:
                        break                    
                    p = parent.param
                    prev_value = parent.value  
                    interact_key = (p, param)
                    if interact_key in interact_shap:
                        interact_score = interact_shap[interact_key].get((prev_value, value), 0)
                        new_score += interact_score
                
                # Create child node
                child = TreeNode(param, value, new_score, depth)
                child.parent = node
                node.children.append(child)
                node.children.sort(key=lambda x: -x. score)
                #bisect.insort(node.children, child, key=lambda x: -x.score)
                heapq.heappush(new_heap, (-new_score, child))
                
        # Pruning: only keep the first beam_width nodes
        heap = heapq.nsmallest(beam_width, new_heap, key=lambda x: x[0])
        for node in new_heap: 
            node[1].mask = True
        for node in heap: 
            node[1].mask = False
        
    return root

    
def extract_top_paths(root, top_n=10):
    paths = []
    
    def dfs(node, path, current_score):
        if not node.children: 
            paths.append((current_score, path.copy()))
            return
        for child in sorted(node.children, key=lambda x: -x.score):
            dfs(child, {**path, child.param: child.value}, child.score)
    
    dfs(root, {}, 0)
    return sorted(paths, key=lambda x: -x[0])[:top_n]


    
# Input score, output color string
def colorgradient(score):
    score_min = -5
    score_max = 5
    color_min = (255,255,255)
    color_max = (24,131,184)
    if score > score_max:
        return '#1883b8'
    elif score < score_min:
        return '#ffffff'
    else:
        color = (int(24+(255-24)*(1 - (score-score_min)/(score_max-score_min))),
                 int(131+(255-131)*(1 - (score-score_min)/(score_max-score_min))),
                 int(184+(255-184)*(1 - (score-score_min)/(score_max-score_min))))

        return "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

    


def generate_dot(root, colorgradient):
    dot = ['digraph Tree {', '  node [shape=circle, style=filled, fixedsize=true, color=none];']
    
    def add_nodes_edges(node):
        if node.mask: 
            return

        dot.append(f'  "{node.depth}-{id(node)}" [label="{node.value}", fillcolor="{colorgradient(node.score)}"];')
    
        if not node.children:
            return
        
        for child in node.children:
            if child.mask: 
                continue

            if node.mask: 
                continue
            label = f"{child.score - node.score:+.2f}"
            dot.append(f'  "{node.depth}-{id(node)}" -> "{child.depth}-{id(child)}" [label="{label}"];')
            add_nodes_edges(child)
    
    add_nodes_edges(root)
    dot.append('}')
    return '\n'.join(dot)




def Analysis_From_SHAP_Folder(shap_dir = 'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/SP19_0116+0109_UseCov/Results/出图/',
                              save_dir = 'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/SP19_0116+0109_UseCov/'):


    shap_dir = shap_dir[:-1]
    suffix = '_SR75_PValue05_S4_vs_S2.csv'
    
    param_order = pd.read_csv(
        rf'{shap_dir}/SHAPSummary{suffix}'
    ).sort_values(by='Mean |SHAP Value|', ascending=False)['Step'].to_list()
    
    main_shap = {
        step: (
            pd.read_csv(
                rf'{shap_dir}/SHAP_{step}{suffix}'
            ).groupby('Step')['SHAP']
            #.median()
            #.sum()
            .quantile(0.05)
            #.agg(lambda x: x[x < x.quantile(0.05)].sum()) #CVaR
            .to_dict() 
        )
        for step in param_order
    }
    
    interact_shap = {
        (step1, step2): {
            tuple(k.split('\n')): v
            for k, v in (
                pd.read_csv(
                    rf'{shap_dir}/SHAPInteraction_{step1}_{step2}{suffix}'
                ).groupby('Combination')['SHAP Interaction Value']
                #.median()
                #.sum()
                .quantile(0.05)
                #.agg(lambda x: x[x < x.quantile(0.05)].sum()) #CVaR
                .to_dict().items()
            )
         }
        for (step1, step2) in pd.read_csv(
            rf'{shap_dir}/SHAPInteractionSummary{suffix}'
        )['Combination'].str.split('\n')
    }    
    print(param_order)
    print(main_shap)
    print(interact_shap)
    # interact shap
    interact_shap.update({
        (p2, p1): {
            (k2, k1): v
            for (k1, k2), v in d.items()
        }
        for (p1, p2), d in interact_shap.items()
    })


    root = beam_search(param_order, main_shap, interact_shap, beam_width=12) # Take the Top 1%
    top_paths = extract_top_paths(root, top_n=12)
    
    top_paths = pd.DataFrame.from_records([x[1] for x in top_paths])
    top_paths.to_csv(save_dir + 'top_paths_VaR95.csv')
    
    dot = generate_dot(root, colorgradient)

    
    # Open a file for writing
    with open(save_dir + 'dot_VaR95.txt', 'w') as file:
        print(dot, file=file)



if __name__ == '__main__': 
    
    Analysis_From_SHAP_Folder(shap_dir = 'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/SP19_0116+0109_UseCov/Results/出图/',
                              save_dir = 'E:/WJW_Code_Hub/SCPDA/Report_Results_Part2/SP19_0116+0109_UseCov/')
    
    


