import shutil
import subprocess
import os
import sys

def generate_cfg(output_path, cpp_path):
    old_working_dir = os.getcwd()
    cpp_file_list = list()
    for file in os.listdir(cpp_path):
        if file.endswith('.cpp'):
            cpp_file_list.append(os.path.join(cpp_path, file))

    for fileidx, cpp_file in enumerate(cpp_file_list):
        print('Processing {} out of {} cpp files'.format(fileidx+1, len(cpp_file_list)))
        os.chdir(cpp_path)
        cpp_filename = os.path.basename(cpp_file)
        cpp_filename_no_ext = os.path.splitext(cpp_filename)[0]
        dot_file_path = os.path.join(output_path, cpp_filename_no_ext)
        if not os.path.exists(dot_file_path):
            os.makedirs(dot_file_path)
        command = ['/home/vagrant/clang+llvm-3.2-x86_64-linux-ubuntu-12.04/bin/clang++', '-emit-llvm', '-S', '{}'.format(cpp_file), '-o',
                   '{}'.format(os.path.join(dot_file_path, cpp_filename_no_ext + '.ll')),
                   '-isystem/home/vagrant/software-gnn/all_merlin_header',
                   '-isystem/home/vagrant/software-gnn/apint_include', '-isystem/usr/include',
                   '-isystem/usr/lib/clang/10.0.0/include',
                   '-isystem/home/vagrant/xfopencv/include/common',
                   '-isystem/home/vagrant/software-gnn/dse_database/ast_kernels', '-isystem/home/vagrant/Vitis_Libraries/vision/L1/include/common', '-include/home/vagrant/software-gnn/apint_include/ap_int.h', '-include/home/vagrant/software-gnn/all_merlin_header/{}_merlin_type_define.h'.format(cpp_filename_no_ext[0:cpp_filename_no_ext.find('_kernel')])]
        res = subprocess.run(command, capture_output=True)
        if res.returncode != 0:
            shutil.rmtree(dot_file_path)
            print("file {} failed".format(cpp_file))
            continue
        os.chdir(dot_file_path)
        command = ['/home/vagrant/clang+llvm-3.2-x86_64-linux-ubuntu-12.04/bin/opt', '--dot-cfg', '{}.ll'.format(cpp_filename_no_ext)]
        res = subprocess.run(command, capture_output=True)
    os.chdir(old_working_dir)


if __name__ == '__main__':
    # output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated_cfg')
    if len(sys.argv)<=1:
        output_path = '/home/vagrant/temp'
    else:
        output_path = sys.argv[1]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if len(sys.argv)<=1:
        cpp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ast_kernels')
    else:
        cpp_path = sys.argv[2]
    generate_cfg(output_path, cpp_path)
