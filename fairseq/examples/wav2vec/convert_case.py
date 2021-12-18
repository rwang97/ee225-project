file = 'dict.txt'
outputfile = 'dict_upper.txt'
with open(outputfile, 'w') as f1:
    with open(file, 'r') as f2:
        for line in f2:
            line = line.upper()
            print(line)
            f1.write(line)