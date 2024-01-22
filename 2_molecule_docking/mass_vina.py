"""
Created on Mon Jan 1 23:03:37 2024

@author: xueyuanyuan0410
"""

import os
import shutil

source_dir_path_1='/home/biosoftware/Autodock/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24'
source_dir_path_2='/home/biosoftware/Autodock/mgltools_x86_64Linux2_1.5.6/bin/pythonsh'
source_dir_path_3='/home/biosoftware/Autodock/autodock_vina_1_1_2_linux_x86/bin'
receptor_path='/home/platform/4_molecular_docking/4_4_vina_docking/receptor/'
ligand_path='/home/platform/4_molecular_docking/4_1_smiles_to_3D/pdb/'
config_path='/home/platform/4_molecular_docking/4_4_vina_docking/config/2fbw_config.txt'
des_dir_path='/home/platform/4_molecular_docking/4_4_vina_docking/vina_docking_dir/'


file_name_list=[]
for filename in os.listdir(ligand_path):
    file_name_list.append(filename)
    basename, ext = os.path.splitext(filename)
    os.system('mkdir '+'/home/platform/4_molecular_docking/4_4_vina_docking/vina_docking_dir/'+basename)

des_dir_paths=[]
for file_1 in os.listdir(des_dir_path):
    for file_2 in os.listdir(source_dir_path_1):
        try:
            shutil.copy(source_dir_path_1+'/'+file_2,des_dir_path+file_1)
        except:
            print(source_dir_path_1+'/'+file_2)
            print(os.path.isfile(source_dir_path_1+'/'+file_2))
    shutil.copy(source_dir_path_2,des_dir_path+file_1+'/pythonsh')
    shutil.copy(config_path,des_dir_path+file_1+'/config.txt')

    for file_7 in os.listdir(source_dir_path_3):
        shutil.copy(source_dir_path_3+'/'+file_7,des_dir_path+file_1)

    for file_3 in os.listdir(receptor_path):
        basename_3, ext_3 = os.path.splitext(file_3)
        shutil.copy(receptor_path+'/'+file_3,des_dir_path+file_1)
       

    for file_4 in os.listdir(ligand_path):
        basename_4, ext_4 = os.path.splitext(file_4)
        if basename_4==file_1:
            shutil.copy(ligand_path+file_4,des_dir_path+file_1)



for file_6 in os.listdir(des_dir_path):
    os.chdir(des_dir_path+file_6)
    os.system('./pythonsh prepare_ligand4.py -l '+file_6+'.pdb')
    os.system('/home/biosoftware/Autodock/autodock_vina_1_1_2_linux_x86/bin/vina --receptor 2fbw-clean.pdbqt --ligand '+file_6+'.pdbqt --config config.txt') 
