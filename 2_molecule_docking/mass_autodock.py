"""
Created on Mon Jan 1 23:03:37 2024

@author: xueyuanyuan0410
"""

import os
import shutil

source_dir_path_1='/home/biosoftware/Autodock/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24'
source_dir_path_2='/home/biosoftware/Autodock/mgltools_x86_64Linux2_1.5.6/bin/pythonsh'
source_dir_path_3='/home/biosoftware/Autodock/x86_64Linux2/'
receptor_path='/home/platform/4_molecular_docking/4_3_autodock_docking/receptor/'
ligand_path='/home/platform/4_molecular_docking/4_1_smiles_to_3D/pdb/pdb_for_unsuccessful/'
des_dir_path='/home/platform/4_molecular_docking/4_3_autodock_docking/autodock_docking_dir/'


file_name_list=[]
for filename in os.listdir(ligand_path):
    file_name_list.append(filename)
    basename, ext = os.path.splitext(filename)
    os.system('mkdir '+'/home/platform/4_molecular_docking/4_3_autodock_docking/autodock_docking_dir/'+basename)

des_dir_paths=[]
for file_1 in os.listdir(des_dir_path):
    for file_2 in os.listdir(source_dir_path_1):
        try:
            shutil.copy(source_dir_path_1+'/'+file_2,des_dir_path+file_1)
        except:
            print(source_dir_path_1+'/'+file_2)
            print(os.path.isfile(source_dir_path_1+'/'+file_2))
    shutil.copy(source_dir_path_2,des_dir_path+file_1+'/pythonsh')

    for file_7 in os.listdir(source_dir_path_3):
        print()
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
    os.system('./pythonsh prepare_gpf4.py -l '+file_6+'.pdbqt'+' -r 2fbw-clean.pdbqt -p npts="40,40,40" -p gridcenter="14.972,16.444,8.694"')
    os.system('./pythonsh prepare_dpf4.py -l '+file_6+'.pdbqt'+' -r 2fbw-clean.pdbqt -p ga_run=20')
    os.system('./autogrid4 -p 2fbw-clean.gpf')
    os.system('./autodock4 -p '+file_6+'_2fbw-clean.dpf')

