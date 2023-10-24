"""
reflexive (e1,r,e1)  Mr = I
symmetric (e1,r1,e2) => (e2,r1,e1) MrMr = I
transitive (e1,r1,e2) & (e2,r1,e3) => (e1,r1,e3) MrMr=Mr
equivalent (e1,r1,e2) <=> (e1,r2,e2) Mr1=Mr2
sub (e1,r1,e2) => (e1,r2,e2) Mr1=Mr2-(M*)
inverse (e1,r1,e2) => (e2,r2,e1) Mr1Mr2 = I
2_hop (e1,r1,e2) & (e2,r2,e3) => (e1,r3,e3) Mr1Mr2 = Mr3
3_hop (e1,r1,e2) & (e2,r2,e3) & (e3,r3,e4) => (e1,r4,e4) Mr1Mr2Mr3 =Mr4
"""
input = r'./a'
output = r'./r'
def extract_reflexive():
    with open('./all_triples.txt', 'r', encoding='utf-8') as fp:
        dat = fp.readlines()
        matrix = []
        for i in range(len(dat)):
            split = dat[i].strip('\n').split("\t")
            matrix.append(split)
        for i in range(len(matrix)):
            if matrix[i][0] == matrix[i][2]:
                with open('./reflexive.txt', 'a', encoding='utf-8') as fout:
                    fout.write(matrix[i][2] + "\t" + matrix[i][1] + "\t" + matrix[i][0] + "\n")


def extract_symmetric():
    with open(input, 'r', encoding='utf-8') as fp:
        dat = fp.readlines()
        matrix = []
        for i in range(len(dat)):
            split = dat[i].strip('\n').split("\t")
            matrix.append(split)
        with open('./symmetric_rule.txt', 'a', encoding='utf-8') as fout:
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    if matrix[i][1] == matrix[j][1] and matrix[i][0] == matrix[j][2] and matrix[i][2] == matrix[j][0]:
                        fout.write(matrix[i][2] + "\t" + matrix[i][1] + "\t" + matrix[i][0] + "\t")
                        fout.write(matrix[i][0] + "\t" + matrix[i][1] + "\t" + matrix[i][2] + "\n")


def extract_transitive():
    with open('../wn18/wn18_triples.train', 'r', encoding='utf-8') as fp:
        dat = fp.readlines()
        matrix = []
        for i in range(len(dat)):
            split = dat[i].strip('\n').split("\t")
            matrix.append(split)
        with open('../wn18/wn18_transitive_rule.tsv', 'a', encoding='utf-8') as fout:
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    if matrix[i][1] == matrix[j][1] and matrix[i][2] == matrix[j][0]:
                        fout.write(matrix[i][1] + "\n")


def extract_equivalent():
    with open('../wn18/wn18_triples.train', 'r', encoding='utf-8') as fp:
        dat = fp.readlines()
        matrix = []
        for i in range(len(dat)):
            split = dat[i].strip('\n').split("\t")
            matrix.append(split)
        with open('../wn18/wn18_equivalent_rule.tsv', 'a', encoding='utf-8') as fout:
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    if matrix[i][0] == matrix[j][0] and matrix[i][2] == matrix[j][2]:
                        fout.write(matrix[i][1] + "\t" + matrix[j][1] + "\n")


def extract_sub():
    with open('../wn18/wn18_triples.train', 'r', encoding='utf-8') as fp:
        dat = fp.readlines()
        matrix = []
        for i in range(len(dat)):
            split = dat[i].strip('\n').split("\t")
            matrix.append(split)
        with open('../wn18/wn18_sub_rule.tsv', 'a', encoding='utf-8') as fout:
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    if matrix[i][0] == matrix[j][0] and matrix[i][2] == matrix[j][2]:
                        fout.write(matrix[i][1] + "\t" + matrix[j][1] + "\n")


def extract_inverse():
    with open('../wn18/wn18_triples.train', 'r', encoding='utf-8') as fp:
        dat = fp.readlines()
        matrix = []
        for i in range(len(dat)):
            split = dat[i].strip('\n').split("\t")
            matrix.append(split)
        with open('../wn18/wn18_inverse_rule.tsv', 'a', encoding='utf-8') as fout:
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    if matrix[i][0] == matrix[j][2] and matrix[i][2] == matrix[j][0]:
                        fout.write(matrix[i][1] + "\t" + matrix[j][1] + "\n")


def extract_2_hop():
    with open('../wn18/wn18_triples.train', 'r', encoding='utf-8') as fp:
        dat = fp.readlines()
        matrix = []
        for i in range(len(dat)):
            split = dat[i].strip('\n').split("\t")
            matrix.append(split)
        with open('../wn18/wn18_2_hop_rule.tsv', 'a', encoding='utf-8') as fout:
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    for k in range(len(matrix)):
                        if matrix[i][2] == matrix[j][0] and matrix[k][0] == matrix[i][0] and matrix[k][2] == matrix[j][
                            2]:
                            fout.write(matrix[i][1] + matrix[j][1] + "\t" + matrix[k][1] + "\n")


def extract_3_hop():
    with open('../wn18/wn18_triples.train', 'r', encoding='utf-8') as fp:
        dat = fp.readlines()
        matrix = []
        for i in range(len(dat)):
            split = dat[i].strip('\n').split("\t")
            matrix.append(split)
        with open('../wn18/wn18_3_hop_rule.tsv', 'a', encoding='utf-8') as fout:
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    for k in range(len(matrix)):
                        for h in range(len(matrix)):
                            if matrix[i][2] == matrix[j][0] and matrix[j][2] == matrix[k][0] \
                                    and matrix[i][0] == matrix[h][0] and matrix[k][2] == matrix[h][2]:
                                fout.write(matrix[i][1] + matrix[j][1] + matrix[k][1] + "\t" + matrix[h][1] + "\n")



