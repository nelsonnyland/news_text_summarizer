import errno
import glob
import os
import string

def main():
    print('Processing txt file...')
    process()

# process the input files
def process():
    try:
        # read input files and process for tokenization
        infile = os.path.dirname(__file__) + '\cnn_dailymail\\train_summaries.txt'
        outfile = os.path.dirname(__file__) + '\cnn_dailymail\\train_summaries_truncated2.txt'
        with open(infile, errors='ignore') as filein, open(outfile, 'w') as fileout:
            lines = filein.readlines()
            # remove header content (start on line 8)
            fileout.writelines(lines[0:50000])
        content = ''
        with open(outfile) as fileInput:
            content = fileInput.read()
            content = ''.join([i if ord(i) < 128 else '' for i in content])
        with open(outfile, 'w') as out:
            out.write(content)
    except IOError as e:
        if e.errno == errno.ENOENT:
            print(os.path(file), ' does not exist')
        if e.errno == errno.EACCES:
            print(os.path(file), ' cannot be read')
        else:
            print(os.path(file), e)

if __name__ == '__main__':
    main()