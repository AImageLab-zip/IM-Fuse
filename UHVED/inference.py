import itertools
import os
import configparser
import argparse
import sys


MODALITIES_img = ['T1', 'T1c', 'T2', 'Flair']

cwd = os.getcwd()
#sys.path.append(os.path.join(cwd,'extensions/'))
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'extensions'))
print(sys.path)



def find_exclusion(path):
    path_folder = os.path.dirname(path)
    name = os.path.basename(path).split('.')[0]
    list_elements = [f for f in os.listdir(path_folder) if os.path.isfile(os.path.join(path_folder, f)) and name in f]
    list_elements.remove(os.path.basename(path))
    list_elements = [k.split('.')[0] for k in list_elements]
    return list_elements


def change_config(net, mod_path, output_mod):

    config = configparser.ConfigParser()
    config.read(['./utils/config.ini'])
    print(config.sections())

    ## Network session
    if net=='u_hemis':
        config.set('NETWORK', 'name', 'u_hemis.u_hemis_net.U_HeMIS')
        config.set('SYSTEM', 'model_dir', './models/u_hemis/')
    else:
        config.set('NETWORK', 'name', 'u_hved.u_hved_net.U_HVED')
        config.set('SYSTEM', 'model_dir', './models/u_hved/')

    ## Provided modality
    choices = [mod_path[k]!=False for k in MODALITIES_img]
    config.set('MULTIMODAL', 'choices', str(tuple(choices)))
    
    # Output
    config.set('MULTIMODAL', 'output_mod', output_mod)

    ## Path
    example_true = choices.index(True)    
    for mod in MODALITIES_img:
        if mod_path[mod]!=False:
            path_folder = os.path.dirname(mod_path[mod])
            name = os.path.basename(mod_path[mod])
            not_names = find_exclusion(mod_path[mod])
            name = name.split('.')[0]
            config.set(mod, 'filename_contains', name[1:])
            config.set(mod, 'path_to_search', path_folder+'/')
            config.set(mod, 'filename_not_contains', str(not_names).replace("'", "")[1:-1])
        else:
            path_folder = os.path.dirname(mod_path[MODALITIES_img[example_true]])
            name = os.path.basename(mod_path[MODALITIES_img[example_true]])
            not_names = find_exclusion(mod_path[MODALITIES_img[example_true]])
            name = name.split('.')[0]
            config.set(mod, 'filename_contains', name[1:])
            config.set(mod, 'path_to_search', path_folder+'/')
            config.set(mod, 'filename_not_contains', str(not_names).replace("'", "")[1:-1])



    with open('./utils/config_temp.ini', 'w') as configfile:
        config.write(configfile)

    name_application = 'U_HeMISApplication' if net=='u_hemis' else  'U_HVEDApplication'

    os.system('net_run inference  -a '+ net+'.application.'+name_application+'  -c ./utils/config_temp.ini --output_postfix ' +name+'_output' )
            








if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Inference using U-HVED or U-HeMIS')
    
#     parser.add_argument('--Warmup', '-w', help='Use Warmup or no', default="False")
    parser.add_argument('--Network', '-n', help='Choice of the network. Either u_hemis or u_hved', default="u_hved")
    parser.add_argument('--T1_input', '-t1', help='Path to the T1 scan', default=False)
    parser.add_argument('--T1c_input', '-t1c', help='Path to the T1c scan', default=False)
    parser.add_argument('--T2_input', '-t2', help='Path to the T2 scan', default=False)
    parser.add_argument('--Flair_input', '-fl', help='Path to the Flair scan', default=False)
    parser.add_argument('--output_mod', '-o', help='Choice of the output: seg, T1, T1c, T2 or Flair', default="seg")

    args = parser.parse_args()
    assert sum([k!=False for k in [args.T1_input, args.T1c_input, args.T2_input, args.Flair_input]]), 'one modality has to been at list provided'
    assert args.output_mod in ['T1', 'T1c', 'T2', 'Flair', 'seg'], "output_mod has to be in ['T1', 'T1c', 'T2', 'Flair', 'seg']"

    modality_path = {'T1': args.T1_input, 'T1c': args.T1c_input, 'T2': args.T2_input, 'Flair':args.Flair_input}

    change_config(args.Network, modality_path, args.output_mod)


