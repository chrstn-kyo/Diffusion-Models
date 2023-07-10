import sys

for line in sys.stdin:
    if line[0] == '!':
        pass
    elif line == '# [:EXPERIMENT SECTION:]\n':
        break
    else:
        sys.stdout.write(line)