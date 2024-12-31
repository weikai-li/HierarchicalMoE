import re


def fix_line(line):
    double_quote_idx = [m.start() for m in re.finditer('"', line)]
    # The first 3 double quotes and the last double quote are legal
    problematic_double_quote_idx = double_quote_idx[3:-1]
    if len(double_quote_idx) == 4:
        return line

    fixed_line = line[0:double_quote_idx[2] + 1] + line[
                                                   double_quote_idx[2] + 1: double_quote_idx[-2] + 1].replace('"', chr(92)+'"') + line[double_quote_idx[-2] + 1:]
    return fixed_line


def fix_broken_cdfg_dot_and_return_lines(dot_file_path):
    dot_file = open(dot_file_path, 'r')
    lines = dot_file.readlines()

    for line_idx in range(len(lines)):
        line = lines[line_idx]
        if line[0] == '"' and ('label' in line):
            # Potential error
            fixed_line = fix_line(line)
            lines[line_idx] = fixed_line
        else:
            continue
    return lines


if __name__ == '__main__':
    exit(0)
