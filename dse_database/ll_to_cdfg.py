import shutil
import subprocess
import os
import sys


def ll_to_cdfg(graph_llvm_ir_tool_path, output_path, ll_path):
    old_working_dir = os.getcwd()
    ll_file_list = list()
    for file in os.listdir(ll_path):
        if file.endswith('.ll'):
            ll_file_list.append(os.path.join(ll_path, file))
    os.chdir(graph_llvm_ir_tool_path)
    for fileidx, ll_file in enumerate(ll_file_list):
        print('Processing {} out of {} ll files'.format(fileidx + 1, len(ll_file)))
        ll_filename = os.path.basename(ll_file)
        ll_filename_no_ext = os.path.splitext(ll_filename)[0]
        dot_file_path = os.path.join(output_path, ll_filename_no_ext)
        command = ['python3', os.path.join(graph_llvm_ir_tool_path, 'graph-llvm-ir'), os.path.join(ll_path, ll_filename)]
        print('about to run command: {}'.format(" ".join(command)))
        res = subprocess.check_output(command)
        if not os.path.exists(dot_file_path):
            os.makedirs(dot_file_path)
        for f in os.listdir(graph_llvm_ir_tool_path):
            print(f)
            if f.endswith('.dot'):
                print('moved dot file to specified output directory')
                shutil.move(os.path.join(graph_llvm_ir_tool_path, f), os.path.join(dot_file_path, f))
    os.chdir(old_working_dir)


if __name__ == '__main__':
    graph_llvm_ir_tool_path = None
    output_path = None
    ll_path = None
    try:
        graph_llvm_ir_tool_path, output_path, ll_path = sys.argv[1], sys.argv[2], sys.argv[3]
    except:
        print('Usage: python3 ll_to_cdfg.py tool_path output_path ll_files_path')
        exit(1)

    ll_to_cdfg(graph_llvm_ir_tool_path, output_path, ll_path)
